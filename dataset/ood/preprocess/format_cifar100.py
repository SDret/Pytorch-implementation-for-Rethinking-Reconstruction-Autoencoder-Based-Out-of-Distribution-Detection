from six.moves import cPickle
import random
import pickle
from easydict import EasyDict
from scipy.io import loadmat
import numpy as np
import os
from PIL import Image


def generate_data_description(save_dir,img_dir,dataset_name,name_list,y):

    dataset = EasyDict()
    dataset.description = dataset_name
    
    dataset.root = img_dir
    dataset.image_name = name_list
    dataset.label = y
    
    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 50000)  # np.array(range(80000))
    dataset.partition.test = np.arange(50000, 60000)  # np.array(range(80000, 90000))

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='bytes')
    fo.close()
    return dict

def conv_data2image(data):
    return np.rollaxis(data.reshape((3,32,32)),0,3)

def get_cifar100(folder):
    train_fname = os.path.join(folder,'train')
    test_fname  = os.path.join(folder,'test')
    data_dict = unpickle(train_fname)
    train_data = data_dict[b'data']
    train_fine_labels = data_dict[b'fine_labels']
    train_coarse_labels = data_dict[b'coarse_labels']

    data_dict = unpickle(test_fname)
    test_data = data_dict[b'data']
    test_fine_labels = data_dict[b'fine_labels']
    test_coarse_labels = data_dict[b'coarse_labels']

    return train_data, np.array(train_coarse_labels), np.array(train_fine_labels), test_data, np.array(test_coarse_labels), np.array(test_fine_labels)


datapath2 = "./data/cifar100/cifar-100-python"
tr_data, _, tr_labels, te_data, _, te_labels = get_cifar100(datapath2)
y = np.concatenate([tr_labels,te_labels],0)

root = 'data/cifar100/images/'
name_list = []
for i in range(tr_data.shape[0]):
    temp = np.reshape(tr_data[i].astype(np.int),[3,32,32]).transpose([1,2,0])
    name = 'tr_' + str(i) + '.png'
    name_list.append(name)
    name = root + name
    temp = Image.fromarray(np.uint8(temp))
    temp.save(name)
    
for i in range(te_data.shape[0]):
    temp = np.reshape(te_data[i].astype(np.int),[3,32,32]).transpose([1,2,0])
    name = 'te_' + str(i) + '.png'
    name_list.append(name)
    name = root + name
    temp = Image.fromarray(np.uint8(temp))
    temp.save(name)
    
np.random.seed(0)
random.seed(0)

save_dir = './data/cifar100'
img_dir = './data/cifar100/images/'
dataset_name = 'cifar100'
generate_data_description(save_dir,img_dir,dataset_name,name_list,y)
