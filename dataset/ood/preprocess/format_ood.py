import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

def generate_data_description(save_dir, img_dir, dataset_name):

    dataset = EasyDict()
    dataset.description = dataset_name
    dataset.root = img_dir
    
    image_name = os.listdir(img_dir)
    img_len = len(image_name)
    print(img_len)
    
    dataset.image_name = image_name
    dataset.partition = np.arange(0, img_len)

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


dataset_name = 'tinyre'

if dataset_name == 'inaturalist':
    save_dir = './data/inaturalist/'
    img_dir = './data/inaturalist/iNaturalist/images'
elif dataset_name == 'isun':
    save_dir = './data/isun/'
    img_dir = './data/isun/iSUN/iSUN_patches'
elif dataset_name == 'lsuncrop':
    save_dir = './data/lsuncrop/'
    img_dir = './data/lsuncrop/LSUN/test'
elif dataset_name == 'lsunre':
    save_dir = './data/lsunre/'
    img_dir = './data/lsunre/LSUN_resize/LSUN_resize/'
elif dataset_name == 'places':
    save_dir = './data/places/'
    img_dir = './data/places/Places/images'
elif dataset_name == 'sun':
    save_dir = './data/sun/'
    img_dir = './data/sun/SUN/images'
elif dataset_name == 'texture':
    save_dir = './data/texture'
    img_dir = './data/texture/images/'
elif dataset_name == 'tinycrop':
    save_dir = './data/tinycrop'
    img_dir = './data/tinycrop/test/'
elif dataset_name == 'tinyre':
    save_dir = './data/tinyre'
    img_dir = './data/tinyre/Imagenet_resize/Imagenet_resize'
    
generate_data_description(save_dir,img_dir,dataset_name)
