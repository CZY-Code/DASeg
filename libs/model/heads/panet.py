"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = {'align': True}
        

    def forward(self, supp_feat, fore_mask, back_mask, qry_feat):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = 2
        n_shots = 1
        n_queries = 1
        batch_size = 4

        img_size = fore_mask.shape[-2:] #[4, 2, 320, 640]
        fts_size = supp_feat.shape[-2:] #[4, 64, 40, 80]

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        for epi in range(batch_size): #Episode
            ###### Extract prototype ######
            # list of tensor[1, 64]
            supp_fg_fts = [self.getFeatures(supp_feat[None, epi],
                           fore_mask[epi, way, None]) for way in range(n_ways)]

            supp_bg_fts = [self.getFeatures(supp_feat[None, epi], back_mask[epi])]
            
            ###### Obtain the prototypes######
            # fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)
            fg_prototypes, bg_prototype = supp_fg_fts, supp_bg_fts
            prototypes = bg_prototype + fg_prototypes

            ###### Compute the distance ######
            #List of [1, 40, 80], length = 3
            dist = [self.calDist(qry_feat[None, epi], prototype) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                #FIXME
                align_loss_epi = self.alignLoss(qry_feat[None, epi], pred, supp_feat[None, epi],
                                                fore_mask[None, epi], back_mask[None, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:]) #[4, 3, 320, 640]
        return output, align_loss / batch_size


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts


    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype


    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        FIXME
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding features for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask[0]), len(fore_mask) #2, 1
        # print(qry_fts.shape) #[1, 64, 40, 80]
        # print(pred.shape) #[1, 3, 40, 80]
        # print(supp_fts.shape) #[1, 64, 40, 80]
        # print(fore_mask.shape) #[1, 2, 320, 640]
        # print(back_mask.shape) #[1, 1, 320, 640]

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)] #3x[1,1,H',W']
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4)) #(1 + Wa) x C
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  #(1 + Wa) x C

        # Compute the support loss
        loss = 0
        for way in range(n_ways): #循环每一类前景
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots): #循环每一次shot
                supp_dist = [self.calDist(supp_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1) #[1,2,40,80]
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:],
                                          mode='bilinear', align_corners=True)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[shot, way], 255,
                                             device=supp_fts.device).long()
                supp_label[fore_mask[shot, way] == 1] = 1
                supp_label[fore_mask[shot, way] == 0] = 0
                # Compute Loss
                loss = loss + F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss
