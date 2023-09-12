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
from libs.dataset.data import VDS, DATA_CONTAINER, multibatch_collate_fn
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
    task_net = FewShotSeg()
    if use_gpu:
        feat_net = feat_net.cuda()
        task_net = task_net.cuda()

    print('Number of feat_net params is: %.2fM' % (sum(p.numel() for p in feat_net.parameters()) / 1e6))
    print('Number of task_net params is: %.2fM' % (sum(p.numel() for p in task_net.parameters()) / 1e6))

    # for p in feat_net.parameters(): #冻结backbone 不更新参数但可以传递梯度
    #     p.requires_grad = False
    
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
        supp_imgs, supp_masks, qry_imgs, qry_masks = data
        if use_cuda:
            supp_imgs = supp_imgs.cuda() #[4,3,320,640]
            supp_masks = supp_masks.cuda() #[4,3,320,640]
            qry_imgs = qry_imgs.cuda() #[4,3,320,640]
            qry_masks = qry_masks.cuda() #[4,3,320,640]
        qry_labels = qry_masks.argmax(dim=1, keepdim=False)  # N x H' x W'

        B, C, H, W = supp_imgs.shape
        imgs_concat = torch.cat([supp_imgs, qry_imgs], dim=0)
        imgs_feats = feat_net(imgs_concat) #长度为3的list 特征图从大到小排列
        
        supp_feat = imgs_feats[0][:B, ...]
        qry_feat  = imgs_feats[0][B:, ...]
        fore_mask = supp_masks[:, 1:, ...]
        # back_mask = 1.0 - fore_mask #此处应该是对每个前景的背景
        back_mask = supp_masks[:, :1, ...] #除开前景后的纯背景
        
        query_pred, align_loss = task_net(supp_feat, fore_mask, back_mask, qry_feat)
        query_loss = criterion(query_pred, qry_labels)
        total_loss = align_loss*align_weight + query_loss*query_weight
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
        

if __name__ == '__main__':
    main()
