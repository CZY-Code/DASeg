import torch
from torch import nn
import torch.nn.functional as F

from libs.model.heads.transformer import MultiHeadAttentionOne


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


class PSPNet(nn.Module):
    def __init__(self, zoom_factor=8):
        super(PSPNet, self).__init__()
        self.bins = [1, 2, 3, 6]
        self.num_classes = 3 #有待商榷
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
        self.classifier = nn.Conv2d(self.bottleneck_dim, self.num_classes, kernel_size=1, bias=False)
        self.transformer = MultiHeadAttentionOne(n_head=1, 
                                                 d_model=self.bottleneck_dim,
                                                 d_k=self.bottleneck_dim, 
                                                 d_v=self.bottleneck_dim,
                                                 dropout=0.1) #0.5

    def freeze_bn(self):
        for m in self.modules():
            if not isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        B, C, H, W = x.size() # #输入的特征大小 [40, 80]
        B //= 2
        output_size = (int(H*self.zoom_factor), int(W*self.zoom_factor))
        
        imgs_feats = self.extract_features(x)
        supp_feats = imgs_feats[:B, ...] #[4, 256, 40, 80]
        qury_feats = imgs_feats[B:, ...] #[4, 256, 40, 80]
        supp_pred = self.classify(supp_feats, output_size) #supp_pred的预测输出

        f_q = F.normalize(qury_feats, dim=1) #[4, 256, 40, 80]
        f_q_reshape = f_q.view(B, self.bottleneck_dim, -1)
        #weight是叶子节点无法无法进行inplace操作所以要.data or .detach()
        weights_cls = self.classifier.weight.detach() #[2, 256, 1, 1]
        weights_cls_reshape = weights_cls.squeeze().unsqueeze(0).expand(B, self.num_classes, weights_cls.shape[1]) # [n_task, 2, c]
        updated_weights_cls = self.transformer(weights_cls_reshape, f_q, f_q)
        qury_pred = torch.matmul(updated_weights_cls, f_q_reshape).view(B, self.num_classes, f_q.shape[-2], f_q.shape[-1])# [n_task, 2, h, w]
        qury_pred = F.interpolate(qury_pred, size=output_size, mode='bilinear', align_corners=True)

        return supp_pred, qury_pred #[4, 2, 80, 160]

    def extract_features(self, x):
        x = self.ppm(x) #[8, 128, 40, 80]
        x = self.bottleneck(x) #[8, 256, 40, 80]
        return x


    def classify(self, features, shape):
        x = self.classifier(features)
        x = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x
