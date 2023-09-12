import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import os
import os.path as osp
import cv2
import time
import argparse
import random
from progress.bar import Bar

from options import OPTION as opt
from libs.utils.logger import Logger, AverageMeter
from libs.dataset.data import VDS, multibatch_collate_fn, COLORS
from libs.model.nets.encoder import Encoder
from libs.model.nets.cosattnet import CosAttNet
from libs.model.heads.panet import FewShotSeg
from libs.utils.iouEval import iouEval

def main():
    random.seed(42)
    torch.cuda.manual_seed(42)
    use_gpu = torch.cuda.is_available() and int(opt.gpu_id)>=0

    print('==> Preparing dataset')
    ds = VDS(train=False, transform=opt.test_transforms)
    testset = data.ConcatDataset([ds] * 1)
    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                  collate_fn=multibatch_collate_fn, drop_last=True)
    print("==> creating model")
    feat_net = Encoder()
    task_net = CosAttNet()
    if use_gpu:
        feat_net = feat_net.cuda()
        task_net = task_net.cuda()

    print('Number of feat_net params is: %.2fM' % (sum(p.numel() for p in feat_net.parameters()) / 1e6))
    print('Number of task_net params is: %.2fM' % (sum(p.numel() for p in task_net.parameters()) / 1e6))

    for p in feat_net.parameters(): #冻结backbone 不更新参数但可以传递梯度
        p.requires_grad = False
    for p in task_net.parameters(): #冻结backbone 不更新参数但可以传递梯度
        p.requires_grad = False
    
    logger = Logger(os.path.join(opt.checkpoint, opt.mode + '_log.txt'), resume=True)
    
    if opt.initial_featNet:
        print('==> Loading from pretrained {}'.format(opt.initial_featNet))
        assert os.path.isfile(opt.initial_featNet), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.initial_featNet)
        feat_net.load_state_dict(checkpoint['state_dict'], strict=True)
    if opt.initial_taskNet:
        print('==> Loading from checkpoint {}'.format(opt.initial_taskNet))
        assert os.path.isfile(opt.initial_taskNet), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.initial_taskNet)
        task_net.load_state_dict(checkpoint['state_dict'], strict=True)

    feat_net.eval()
    task_net.eval()
    metrics = test(testloader, feat_net=feat_net,
                   task_net=task_net, use_cuda=use_gpu)
    print(metrics)
    logger.close()
        

def test(testloader, feat_net, task_net, use_cuda):
    iouEval_drivable = iouEval(nClasses=3, ignoreIndex=-1)
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, data in enumerate(testloader): #循环iter
        start = time.time()
        supp_imgs, supp_masks, qury_imgs, qury_masks = data
        if use_cuda:
            supp_imgs = supp_imgs.cuda()   #[4,3,320,640]
            supp_masks = supp_masks.cuda() #[4,3,320,640]
            qury_imgs = qury_imgs.cuda()   #[4,3,320,640]
            qury_masks = qury_masks.cuda() #[4,3,320,640]
        qury_labels = qury_masks.argmax(dim=1, keepdim=False)  # N x H' x W'
        supp_labels = supp_masks.argmax(dim=1, keepdim=False)  # N x H' x W'

        imgs_concat = torch.cat([supp_imgs, qury_imgs], dim=0)
        with torch.no_grad():
            imgs_feats = feat_net(imgs_concat) #长度为3的list 特征图从大到小排列
            supp_pred, qury_pred = task_net(imgs_feats[0], supp_masks, qury_masks)
        end = time.time()

        # show_pred(qury_imgs, qury_pred, qury_labels, batch_idx)
        qury_pred = F.softmax(qury_pred, dim=1).argmax(1).unsqueeze(1)
        supp_pred = F.softmax(supp_pred, dim=1).argmax(1).unsqueeze(1)
        da_tp, da_fp, da_fn = iouEval_drivable.addBatch(qury_pred, qury_masks)
        da_tp, da_fp, da_fn = iouEval_drivable.addBatch(supp_pred, supp_masks)
        # show_pred(supp_imgs, supp_pred, supp_labels)
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=end-start
            )
        bar.next()
    bar.finish()
    metrics, *_ = iouEval_drivable.getIoU()
    return metrics

def show_pred(supp_imgs, supp_pred, supp_labels, batch_idx):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(np.float32)
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3]).astype(np.float32)
    for img, pred, label in zip(supp_imgs, supp_pred, supp_labels):
        img = img.detach().clone().permute(1, 2, 0).cpu().numpy()#.astype(np.uint8)
        img = ((img * std + mean) * 255.0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = pred.detach().clone().softmax(0).argmax(0).cpu().numpy()
        seg_pred = np.zeros((320, 640, 3), dtype=np.uint8)
        for k in range(1, pred.max()+1):
            seg_pred[pred==k, :] = COLORS[k-1] #sample['palette'][(k*3):(k+1)*3]
        im_pred = cv2.addWeighted(img, 0.8, seg_pred, 0.2, 0, dtype = -1)
        cv2.imshow('im_pred', im_pred)

        label = label.detach().clone().cpu().numpy()
        seg_label = np.zeros((320, 640, 3), dtype=np.uint8)
        for k in range(1, label.max()+1):
            seg_label[label==k, :] = COLORS[k-1] #sample['palette'][(k*3):(k+1)*3]
        im_label = cv2.addWeighted(img, 0.8, seg_label, 0.2, 0, dtype = -1)
        cv2.imshow('im_label', im_label)
        cv2.waitKey(0)
        # cv2.imwrite('./output/label/{}.jpg'.format(batch_idx), im_label)
        cv2.imwrite('./output/pred/{}.jpg'.format(batch_idx), im_pred)


if __name__ == '__main__':
    main()
