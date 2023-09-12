from tkinter.messagebox import NO
import torch
import torch.nn as nn

from libs.model.backbones import *
from libs.model.necks import *
from libs.model.heads import *

class Encoder(nn.Module): #resnet with FPN
    def __init__(self, cfg=None):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.backbone = ResNetWrapper()
        self.neck = FPN(in_channels=[128, 256, 512],
                        out_channels=64,
                        num_outs=4) #3

    def forward(self, batch):
        fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)
        fea = self.neck(fea)
        return fea