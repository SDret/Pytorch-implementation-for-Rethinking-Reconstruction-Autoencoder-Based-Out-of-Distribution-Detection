import math

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from einops import rearrange,repeat,reduce
from torch.nn.parameter import Parameter
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
 
    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

class ILCE(nn.Module):
    def __init__(self, c_in, c_h, c_h_2, c_out):
        super(ILCE, self).__init__()

        self.linear_1 = nn.Linear(c_in, c_h)
        self.act_1 = Swish()
        torch.nn.init.xavier_uniform(self.linear_1.weight)
        
        self.linear_2 = nn.Linear(c_h, c_h_2)
        self.act_2 = Swish()
        torch.nn.init.xavier_uniform(self.linear_2.weight)
        
        self.linear_3 = nn.Linear(c_h_2, c_out)
        torch.nn.init.xavier_uniform(self.linear_3.weight)
        
    def forward(self, x):
        
        out = self.linear_3(self.act_2(self.linear_2(self.act_1(self.linear_1(x)))))
        
        return out
    

class Classifier(nn.Module):

    def __init__(self, c_in, nattr):
        super(Classifier, self).__init__()
        
        self.logits = nn.Linear(c_in, nattr)
        torch.nn.init.xavier_uniform(self.logits.weight)

    def forward(self, x):
        
        logits = self.logits(x)
        
        return logits

class Network(nn.Module):
    def __init__(self, backbone, classifier, decoder_1, decoder_2):
        super(Network, self).__init__()

        self.backbone = backbone  
        
        if decoder_1 != None:
            for p in self.backbone.parameters():
                p.requires_grad=False
            
        self.classifier = classifier
        self.decoder_1 = decoder_1
        self.decoder_2 = decoder_2
        
    def forward(self, x):
        
        if self.decoder_1 == None:
            
            x = self.backbone(x)
            x = self.classifier(x)
            
            return x
        
        else:
            
            low_1 = self.backbone(x)
            x_1 = self.classifier(low_1)
            low_2 = x_1/100
            
            softmax_t = F.softmax(low_2,dim=-1)
            rec_1 = self.decoder_1(x_1)
            rec_2 = self.decoder_2(softmax_t)
            
            return x_1, low_1, low_2, rec_1, rec_2