import glob
import os
import pickle
import numpy as np
import torch.utils.data as data
from PIL import Image
from tools.function import get_pkl_rootpath
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import torch

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ood_dataloader(data.Dataset):

    def __init__(self, dataset_name, split, transform=None):

        assert dataset_name in ['cifar10', 'cifar100', 'imagenet1k', 'inaturalist','isun','lsuncrop','lsunre','places','sun','texture','tinycrop','tinyre'], \
            f'dataset name {dataset_name} is not exist'
        
        if dataset_name not in ['cifar10','cifar100','imagenet1k']:
            
            data_path = get_pkl_rootpath(dataset_name)
            print("which pickle", data_path)
            
            dataset_info = pickle.load(open(data_path, 'rb+'))
            self.dataset = dataset_name
            self.root_path = dataset_info.root
            self.transform = transform
            
            self.attr_num = 100
            
            img_id = dataset_info.image_name
            self.img_idx = dataset_info.partition
            self.img_id = [img_id[i] for i in self.img_idx]
            self.img_num = self.img_idx.shape[0]
            self.label = np.zeros(self.img_num)
                
        else: 
            data_path = get_pkl_rootpath(dataset_name)
            print("which pickle", data_path)

            dataset_info = pickle.load(open(data_path, 'rb+'))
            self.dataset = dataset_name
            self.root_path = dataset_info.root
            self.transform = transform
            
            if dataset_name == 'cifar100':
                self.attr_num = 100
            elif dataset_name == 'cifar10':
                self.attr_num = 10
            elif dataset_name == 'imagenet1k':
                self.attr_num = 1000 
            
            assert split in dataset_info.partition.keys(), f'split {split} is not exist'
            img_id = dataset_info.image_name
            self.img_idx = dataset_info.partition[split]
            self.img_id = [img_id[i] for i in self.img_idx]
            self.label = dataset_info.label[self.img_idx]
            self.img_num = self.img_idx.shape[0]
            
    def __getitem__(self, index):
        
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]    
        imgpath = os.path.join(self.root_path, imgname)
        
        img = Image.open(imgpath)
        
        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            img = PIL.Image.merge("RGB", (r,g,b))
        elif img.mode != 'RGB':
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
            
        return img, gt_label

    def __len__(self):
        return self.img_num