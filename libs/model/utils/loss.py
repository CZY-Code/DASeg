import torch
import torch.nn as nn
import torch.nn.functional as F

class DASegCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            # weight=torch.tensor([1.0, len(back_pix[0]) / len(target_pix[0])]).cuda(),
            ignore_index=255)
        # self.loss = nn.NLLLoss(ignore_index=255)
    
    def forward(self, preds, labels):
        # preds = F.log_softmax(preds)
        return self.loss(preds, labels)