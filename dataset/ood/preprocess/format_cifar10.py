from __future__ import print_function

from six.moves import cPickle as pickle
import numpy as np
import os
import platform
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat
from PIL import Image

np.random.seed(0)
random.seed(0)

def generate_data_description(save_dir,img_dir,dataset_name,name_list,y):

    dataset = EasyDict()
    dataset.description = dataset_name
    
    dataset.root = img_dir
    dataset.image_name = name_list
    dataset.label = y
    
    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 50000) 
    dataset.partition.test = np.arange(50000, 60000) 

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

def load_pickle(f):
    version = platform.python_version_tuple() 
    if version[0] == '2':
        return  pickle.load(f) 
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)   
    X = datadict['data']        
    Y = datadict['labels']      
    
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = [] # list
  ys = []
  
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y

  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

xtr,ytr,xte,yte = load_CIFAR10('data/cifar10/cifar-10-batches-py')

root = 'data/cifar10/images/'
name_list = []
for i in range(len(xtr)):
    temp = np.reshape(xtr[i].astype(np.int),[32,32,3])
    name = 'tr_' + str(i) + '.png'
    name_list.append(name)
    name = root + name
    temp = Image.fromarray(np.uint8(temp))
    temp.save(name)
    
for i in range(len(xte)):
    temp = np.reshape(xte[i].astype(np.int),[32,32,3])
    name = 'te_' + str(i) + '.png'
    name_list.append(name)
    name = root + name
    temp = Image.fromarray(np.uint8(temp))
    temp.save(name)
    
y = np.concatenate([ytr,yte],0)

save_dir = './data/cifar10'
img_dir = './data/cifar10/images/'
dataset_name = 'cifar10'
generate_data_description(save_dir,img_dir,dataset_name,name_list,y)