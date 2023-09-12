import torch
from torch import nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import seaborn as sns
from time import time
import pandas as pd

from libs.model.heads.transformer import MultiHeadAttentionOne
from libs.model.utils.position_encoding import PositionEmbeddingSine


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True))
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class CosAttNet(nn.Module):
    def __init__(self, zoom_factor=8):
        super(CosAttNet, self).__init__()
        self.bins = [1, 2, 3, 6]
        self.nClasses = 3
        assert zoom_factor in [1, 2, 4, 8] 
        self.zoom_factor = zoom_factor
        self.bottleneck_dim = 256

        self.fea_dim = 64 #512
        self.ppm = PPM(self.fea_dim, int(self.fea_dim/len(self.bins)), self.bins)
        self.fea_dim *= 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.fea_dim, self.bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.classifier = nn.Conv2d(self.bottleneck_dim, self.nClasses, kernel_size=1, bias=False) #没用
        
        self.transformer = MultiHeadAttentionOne(n_head=1, 
                                                 d_model=self.bottleneck_dim,
                                                 d_k=self.bottleneck_dim, 
                                                 d_v=self.bottleneck_dim,
                                                 dropout=0.1, #0.5 元学习最忌讳对单一样本的过拟合
                                                 attention='CosineSimilartyAttention') #
        self.position = PositionEmbeddingSine(num_pos_feats=self.bottleneck_dim//2)

    def freeze_bn(self):
        for m in self.modules():
            if not isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x, supp_labels, qury_labels):
        B, C, H, W = x.size() # #输入的特征大小 [40, 80]
        B //= 2
        output_size = (int(H*self.zoom_factor), int(W*self.zoom_factor))
        
        imgs_feats = self.extract_features(x)
        #TODO:should add positional embedding in feature
        pos_embed = self.position(imgs_feats)
        imgs_feats += pos_embed

        supp_feats = imgs_feats[:B, ...] #[4, 256, 40, 80]
        qury_feats = imgs_feats[B:, ...] #[4, 256, 40, 80]
        # origin support to label
        # origin_supp_pred = self.classifier(supp_feats)
        # origin_qury_pred = self.classifier(qury_feats)

        #support to query
        supp_prototype = self.getFeatures(supp_feats, supp_labels)
        # cond_qurytype = self.transformer(supp_prototype, qury_feats, qury_feats) #[4, 3, 256]
        cond_qurytype = supp_prototype
        if not self.training:
            # self.visualization_TSNE(cond_qurytype, qury_feats, qury_labels)
            self.T_sne_visual(cond_qurytype, qury_feats, qury_labels)

        qury_pred = self.classify(cond_qurytype, qury_feats) #[4,3,40,80]
        qury_pred = F.interpolate(qury_pred, size=output_size, mode='bilinear', align_corners=True)
        #query to support
        pred_labels = self.makeOnehot(qury_pred)
        qury_prototype = self.getFeatures(qury_feats, pred_labels)
        cond_supptype = self.transformer(qury_prototype, supp_feats, supp_feats)
        # cond_supptype = qury_prototype
        # if not self.training:
        #     self.T_sne_visual(cond_supptype, supp_feats, supp_labels)
        
        supp_pred = self.classify(cond_supptype, supp_feats)
        supp_pred = F.interpolate(supp_pred, size=output_size, mode='bilinear', align_corners=True)

        return supp_pred, qury_pred #[4, 3, 320, 640]

    def extract_features(self, x):
        x = self.ppm(x) #[8, 128, 40, 80]
        x = self.bottleneck(x) #[8, 256, 40, 80]
        return x
    
    def classify(self, prototype, feat, scaler=20): #sclar为什么是20？
        '''
        prototype expected shape B x N x C
        feat expected shape B x C x H' x W'
        output expected shape B x N x H' x W'
        '''
        prototype_re = prototype[:, :, None, None, :]
        feat_reshape = feat.permute(0, 2, 3, 1).unsqueeze(1)
        output = F.cosine_similarity(prototype_re, feat_reshape, dim=-1) * scaler #[4,3,40,80]
        return output

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling
        Args:
            fts: input features, expect shape: B x C x H' x W'
            mask: binary mask, expect shape: B x N x H x W
        return:
            prototype: expect shape: B x N x C
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts.unsqueeze(1) * mask.unsqueeze(2), dim=(3, 4)) \
                     / (mask.unsqueeze(2).sum(dim=(3, 4)) + 1e-5) #[4, 3, 256]
        return masked_fts

    def makeOnehot(self, x):
        x_pred = x.detach().softmax(1).argmax(1).unsqueeze(1) #detach
        x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))
        if x.is_cuda:
            x_onehot = x_onehot.cuda()
        x_onehot.scatter_(1, x_pred, 1).float()
        return x_onehot


    def visualization_TSNE(self, prototype, feat, labels):
        label = labels.detach().clone()
        label = F.interpolate(label, size= feat.shape[-2:], mode='bilinear', align_corners=True)
        label = label.argmax(1).flatten(1).transpose(0,1).cpu().numpy() #[3200, 1]
        
        source = feat.detach().clone().transpose(0,1).flatten(1).transpose(0,1).cpu().numpy()
        target = prototype.detach().clone().view(-1, prototype.shape[-1]).cpu().numpy()
        total_sample = np.concatenate([target, source], axis=0)
        tsne = TSNE(n_components=2, init='pca')
        # res = tsne.fit_transform(source) #(3200, 2)
        # tgt = tsne.fit_transform(target) #(3, 2)
        result = tsne.fit_transform(total_sample) #[3203, 256]
        self.plot_embedding(result, label, 'metric space')
        plt.show()

    def plot_embedding(self, result, Y, title):
        x_min, x_max = np.min(result, 0), np.max(result, 0)
        data = (result - x_min) / (x_max - x_min)
        # center = (tgt - x_min) / (x_max - x_min)
        source = data[3:]
        center = data[:3]
        plt.figure(figsize=(8,6))
        ax = plt.subplot(111)
        cSource = ['gray', 'cyan', 'yellow']
        LSource = ['B', 'D', 'A']
        cCenter = ['black', 'blue', 'red']
        LCenter = ['PB', 'PD', 'PA']
        # for idx, label in enumerate(Y[:, 0]):
        #     ax.scatter(source[idx, 0], source[idx, 1], c = cSource[label], marker = 'o', label = LSource[label], s = 16, alpha=0.2)
        # for idx in range(len(center)):
        #     ax.scatter(center[idx, 0], center[idx, 1], c = cCenter[idx], marker="^" , label = LCenter[idx], s = 64, alpha=1.0)
        ax.scatter(source[:, 0], source[:, 1], c = [cSource[Y[i, 0]] for i in range(Y.shape[0])], marker = 'o', label = "Feature", s = 16, alpha=0.2)
        ax.scatter(center[:, 0], center[:, 1], c = cCenter, marker="^" , label = "Prototype", s = 64, alpha=1.0)

        plt.xticks([])
        plt.yticks([])
        # ax.legend(loc='upper left', frameon=False)
        plt.legend(loc='upper left', title='')
        plt.ylabel('')
        plt.xlabel('')
        return

    def T_sne_visual(self, cond_qurytype, qury_feats, qury_labels):
        # t_classes = ['QB','QD','QA']
        # s_classes = ['B', 'D', 'A']
        t_classes = ['xkcd:grey', 'xkcd:scarlet', 'xkcd:grass green']
        s_classes = ['xkcd:silver', 'xkcd:bright pink', 'xkcd:bright green']
        labels = qury_labels.detach().clone()
        labels = F.interpolate(labels, size= qury_feats.shape[-2:], mode='bilinear', align_corners=True)
        labels = labels.argmax(1).flatten(1).transpose(0,1).cpu().numpy() #[3200, 1]
        source = qury_feats.detach().clone().transpose(0,1).flatten(1).transpose(0,1).cpu().numpy() #[3200, 256]
        target = cond_qurytype.detach().clone().view(-1, cond_qurytype.shape[-1]).cpu().numpy() #[3, 256]
        # total_sample = np.concatenate([target, source], axis=0)
        total_sample = []
        total_label = []
        for label, feat in zip(t_classes, target):
            total_sample.append(feat)
            total_label.append(label)
        for label, feat in zip(labels, source):
            total_sample.append(feat)
            total_label.append(s_classes[label[0]])

        # self.t_sne(np.array(total_sample), total_label, title=f'Dataset visualize result\n')
        self.t_sne_3D(np.array(total_sample), total_label, title=f'Dataset visualize result\n')

    def t_sne_3D(self, data, label, title): #prototype, feat, labels       
        print('starting T-SNE process')
        data = TSNE(n_components=3, init='pca').fit_transform(data)
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min) #[3203, 3]
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])  # 转换成df表
        df.insert(loc=1, column='label', value=label)
        print('Finished')
        # 绘图
        markers = { "QB": '^', 
                    "QD": '^',
                    "QA": '^',
                    "B": "o",
                    "D": "o",
                    "A": "o"}
        fig = plt.figure(figsize=(5, 4))
        ax = Axes3D(fig)
        ax.scatter(df.x[3::5],df.y[3::5],df.z[3::5], alpha=0.8, s=15, marker='^', c=df.label[3::5]) 
        ax.scatter(df.x[:3],df.y[:3],df.z[:3], alpha=1.0, s=50, marker='o', c=df.label[:3])
        self.draw_surface(data[1:3], radius=0.15, colors=label[1:3], ax=ax)
        #设置背景平面的颜色
        # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.set_xlabel('X', {'size': 14})
        ax.set_ylabel('Y', {'size': 14})
        ax.set_zlabel('Z', {'size': 14})
        ax.tick_params(axis='x',labelsize=14)
        ax.tick_params(axis='y',labelsize=14)
        ax.tick_params(axis='z',labelsize=14)
        #设置坐标刻度数量
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.zaxis.set_major_locator(plt.MaxNLocator(3))
        ax.grid(None)
        plt.show()

    def t_sne(self, data, label, title): #prototype, feat, labels        
        print('starting T-SNE process')
        start_time = time()
        data = TSNE(n_components=2, init='pca').fit_transform(data)
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        df = pd.DataFrame(data, columns=['x', 'y'])  # 转换成df表
        df.insert(loc=1, column='label', value=label)
        end_time = time()
        print('Finished')

        # 绘图
        markers = { "QB": '^', 
                    "QD": '^',
                    "QA": '^',
                    "B": "o",
                    "D": "o",
                    "A": "o"}
        # cmap = sns.palplot(sns.diverging_palette(145,280, s=85, l=25, n=3)) #调出调色板
        # sns.set_palette(palette=sns.diverging_palette(145,280, s=85, l=25, n=3))
        Scolors = ['silver', 'royalblue', 'mediumseagreen']
        Ccolors = ['gray', 'darkblue', 'darkgreen']
        # sns.set_palette(palette=colors)
        sns.scatterplot(x='x', y='y', hue='label', hue_order = ['B','D','A'], palette=Scolors, 
                        style = 'label', markers=markers, s=20, data=df[3:], alpha=0.7)
        sns.scatterplot(x='x', y='y', hue='label', hue_order = ['QB','QD','QA'], palette=Ccolors, 
                        style='label', markers=markers, s=80, data=df[:3]) 
        self.set_plt(start_time, end_time, title)
        # plt.savefig('1.jpg', dpi=400)
        plt.show()

    def set_plt(self, start_time, end_time, title):
        # plt.title(f'{title} time consume:{end_time - start_time:.3f} s')
        # plt.title('The distribution in the metric space')
        plt.legend(loc='upper right')
        plt.ylabel('')
        plt.xlabel('')
        plt.xticks([])
        plt.yticks([])

    def draw_surface(self, centers, radius, colors, ax:Axes3D):
        # data
        for center, color in zip(centers, colors):
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
            ax.plot_surface(x, y, z, alpha=0.15, rstride=4, cstride=4, color=color)