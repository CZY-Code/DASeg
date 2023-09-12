import torch
import torch.nn as nn
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
from libs.utils.utility import save_checkpoint
from libs.dataset.data import VDS, multibatch_collate_fn, COLORS
from libs.model.nets.encoder import Encoder
from libs.model.nets.pspnet import PSPNet
from libs.model.heads.panet import FewShotSeg
from libs.model.utils.loss import DASegCriterion

def main():
    random.seed(42)
    torch.cuda.manual_seed(42)
    use_gpu = torch.cuda.is_available() and int(opt.gpu_id)>=0

    print('==> Preparing dataset')
    ds = VDS(transform=opt.test_transforms)
    trainset = data.ConcatDataset([ds] * opt.datafreq)
    trainloader = data.DataLoader(trainset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers,
                                  collate_fn=multibatch_collate_fn, drop_last=True)
    print("==> creating model")
    feat_net = Encoder()
    task_net = PSPNet() #FewShotSeg()
    if use_gpu:
        feat_net = feat_net.cuda()
        task_net = task_net.cuda()

    print('Number of feat_net params is: %.2fM' % (sum(p.numel() for p in feat_net.parameters()) / 1e6))
    print('Number of task_net params is: %.2fM' % (sum(p.numel() for p in task_net.parameters()) / 1e6))

    for p in feat_net.parameters(): #冻结backbone 不更新参数但可以传递梯度
        p.requires_grad = False
    
    # FIXME define criterion
    criterion = DASegCriterion()

    if opt.solver == 'sgd':
        params = [{"params": feat_net.parameters(), 'lr': opt.learning_rate},
                    {"params": task_net.parameters(), 'lr': opt.learning_rate}]
        optimizer = optim.SGD(params, momentum=opt.momentum[0], weight_decay=opt.weight_decay)

    elif opt.solver == 'adam':
        params = [{'params': feat_net.parameters(), 'lr': opt.learning_rate},
                    {'params': task_net.parameters(), 'lr': opt.learning_rate}]
        optimizer = optim.Adam(params, 
                                betas=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.solver == 'adamW':
        params = [{'params': feat_net.parameters(), 'lr': opt.learning_rate},
                    {'params': task_net.parameters(), 'lr': opt.learning_rate}]
        optimizer = optim.AdamW(params, betas=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise TypeError('unkown solver type %s' % opt.solver)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500*opt.epochs)

    minloss = float('inf')

    opt.checkpoint_Feat = osp.join(osp.join(opt.checkpoint, 'featModel'))
    opt.checkpoint_Task = osp.join(osp.join(opt.checkpoint, 'taskModel'))
    if not osp.exists(opt.checkpoint_Feat):
        os.makedirs(opt.checkpoint_Feat)
    if not osp.exists(opt.checkpoint_Task):
        os.makedirs(opt.checkpoint_Task)
    

    if opt.initial_featNet:
        print('==> Init from checkpoint {}'.format(opt.initial_featNet))
        assert os.path.isfile(opt.initial_featNet), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.initial_featNet)
        feat_net.load_state_dict(checkpoint['state_dict'], strict=True)
    elif opt.resume_featNet:
        print('==> Resuming from pretrained {}'.format(opt.resume_featNet))
        assert os.path.isfile(opt.resume_featNet), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.resume_featNet)
        feat_net.load_state_dict(checkpoint['state_dict'], strict=True)

    logger = Logger(os.path.join(opt.checkpoint, opt.mode + '_log.txt'), resume=True)

    if opt.initial_taskNet:
        print('==> Init from checkpoint {}'.format(opt.initial_taskNet))
        assert os.path.isfile(opt.initial_taskNet), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.initial_taskNet)
        task_net.load_state_dict(checkpoint['state_dict'], strict=True)
    elif opt.resume_taskNet:
        print('==> Resuming from checkpoint {}'.format(opt.resume_taskNet))
        assert os.path.isfile(opt.resume_taskNet), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.resume_taskNet)
        minloss = checkpoint['minloss']
        start_epoch = checkpoint['epoch']
        task_net.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    # print(feat_net.backbone.model.conv1.weight.requires_grad)
    logger.set_items(['Epoch', 'LR', 'Train Loss'])

    start_epoch = 0
    for epoch in range(start_epoch, opt.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, optimizer.param_groups[0]['lr']))
        feat_net.train()
        task_net.train()
        criterion.train()
        train_loss = train(trainloader,
                            feat_net=feat_net,
                            task_net=task_net,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            use_cuda=use_gpu)
        # append logger file
        logger.log(epoch + 1, opt.learning_rate, train_loss)
        is_best = train_loss <= minloss
        minloss = min(minloss, train_loss)
        
        if ((epoch + 1) % opt.epoch_per_test == 0) or (is_best):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': task_net.state_dict(),
                'loss': train_loss,
                'minloss': minloss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, epoch + 1, is_best, checkpoint=opt.checkpoint_Task, filename=opt.mode)
    logger.close()
    print('minimum loss:', minloss)
        

def train(trainloader, feat_net, task_net, criterion, optimizer, scheduler, epoch, use_cuda):
    align_weight = 4.0
    query_weight = 6.0
    loss = AverageMeter()
    bar = Bar('Processing', max=len(trainloader))
    
    optimizer.zero_grad()
    for batch_idx, data in enumerate(trainloader): #循环iter
        start = time.time()
        total_loss = 0.0
        supp_imgs, supp_masks, qury_imgs, qury_masks = data
        if use_cuda:
            supp_imgs = supp_imgs.cuda() #[4,3,320,640]
            supp_masks = supp_masks.cuda() #[4,3,320,640]
            qury_imgs = qury_imgs.cuda() #[4,3,320,640]
            qury_masks = qury_masks.cuda() #[4,3,320,640]
        qury_labels = qury_masks.argmax(dim=1, keepdim=False)  # N x H' x W'
        supp_labels = supp_masks.argmax(dim=1, keepdim=False)  # N x H' x W'

        B, C, H, W = supp_imgs.shape
        imgs_concat = torch.cat([supp_imgs, qury_imgs], dim=0)
        imgs_feats = feat_net(imgs_concat) #长度为3的list 特征图从大到小排列
        supp_pred, qury_pred = task_net(imgs_feats[0])
        # show_pred(supp_imgs, supp_pred, supp_labels)
        supp_loss = criterion(supp_pred, supp_labels)
        qery_loss = criterion(qury_pred, qury_labels)
        total_loss = supp_loss*align_weight + qery_loss*query_weight
        total_loss /= B

        # record loss
        if isinstance(total_loss, torch.Tensor) and total_loss.item() > 0.0:
            loss.update(total_loss.item(), 1)

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.5f}'.format(
                batch=batch_idx + 1,
                size=len(trainloader),
                data=end-start,
                loss=total_loss.item() #loss.avg
            )
        print('-'*20 + str(loss.avg))
        bar.next()
    bar.finish()

    return loss.avg

def show_pred(supp_imgs, supp_pred, supp_labels):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(np.float32)
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3]).astype(np.float32)
    for img, pred, label in zip(supp_imgs, supp_pred, supp_labels):
        img = img.detach().clone().permute(1, 2, 0).cpu().numpy()#.astype(np.uint8)
        img = ((img*std+mean)*255.0).astype(np.uint8)
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


if __name__ == '__main__':
    main()
