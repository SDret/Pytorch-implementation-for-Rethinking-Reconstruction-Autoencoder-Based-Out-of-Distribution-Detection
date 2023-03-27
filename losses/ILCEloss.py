import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight

class ILCE_loss(nn.Module):
    def __init__(self, loss_weight):
        super(ILCE_loss,self).__init__()
        
        self.loss_weight = loss_weight
        self.cls = nn.CrossEntropyLoss()
        
    def forward(self, logits, targets, x_0, x_1, rec_0, rec_1):
        
        loss_cls = self.cls(logits,targets)
        
        if self.loss_weight != 0.:
            loss_rec_0 = torch.mean(torch.sum((x_0 - rec_0)**2,dim=-1))
            loss_rec_1 = torch.mean(torch.sum((x_1 - rec_1)**2,dim=-1))
            loss_cls = self.loss_weight * loss_cls
            
            return loss_cls, loss_rec_0, loss_rec_1
        else:
            return loss_cls