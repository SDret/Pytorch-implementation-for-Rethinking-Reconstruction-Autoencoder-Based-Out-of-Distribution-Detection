import numpy as np
from easydict import EasyDict
import torch


def get_cls_metrics(output, target, topk=(1,)):
   
    result = EasyDict()
   
    output = torch.tensor(output,dtype=torch.float)
    target = torch.tensor(target,dtype=torch.float)
    
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).numpy())
    
    result.acc = res[0] if maxk == 1 else res
    
    return result
