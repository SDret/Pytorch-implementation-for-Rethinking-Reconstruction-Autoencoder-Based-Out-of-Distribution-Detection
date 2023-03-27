import random
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from dataset.autoaug import AutoAugment

class RandomCrop(object):

    def __init__(self, crop_shape, padding = None, p = 0.5):

        self.crop_shape = crop_shape
        self.p = p
        self.padding = padding
        
    def __call__(self, batch_):
    
        if random.uniform(0, 1) < self.p:
            
            batch = np.array(batch_).copy()
            oshape = np.shape(batch)
            oshape = (oshape[0] + 2 * self.padding, oshape[1] + 2 * self.padding)
            npad = ((self.padding, self.padding), (self.padding, self.padding), (0, 0))
            batch = np.lib.pad(batch, pad_width=npad,mode='constant', constant_values=0)
            nh = random.randint(0, oshape[0] - self.crop_shape[0])
            nw = random.randint(0, oshape[1] - self.crop_shape[1])
            batch = batch[nh:nh + self.crop_shape[0],nw:nw + self.crop_shape[1]]
            
            return Image.fromarray(batch.astype('uint8')).convert('RGB')
        
        else:
            
            return batch_

def get_transform(cfg):
    
    if cfg.DATASET.NAME == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]
    elif cfg.DATASET.NAME == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif cfg.DATASET.NAME == 'imagenet1k':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        
    normalize = T.Normalize(mean=mean, std=std)

    if cfg.DATASET.TYPE == 'cifar':
        width = 32
    
    if cfg.NAME == 'ID':
        train_transform = T.Compose([
            RandomCrop([32,32],4,0.5),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            normalize,
        ])
    else:
        train_transform = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            normalize,
        ])

    valid_transform = T.Compose([
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
