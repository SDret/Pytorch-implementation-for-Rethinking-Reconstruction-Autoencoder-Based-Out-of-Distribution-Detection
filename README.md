# Pytorch-implementation-for-Rethinking-Reconstruction-Autoencoder-Based-Out-of-Distribution-Detection
This is the public implementation of CVPR2022 paper 'Rethinking Reconstruction Autoencoder-Based Out-of-Distribution Detection'.

This code comes late since the author focused on otehr area and now turns back to the problem of Out-of-distribution. The original benchmark result in paper was produced in Tensorflow 1.14.0, and this codebase is a pytorch version of re-implementing it. Thus, there might be few of differences regarding the original result. Specifically, The result on iSUN is slightly better than that in paper, while lsun-crop is not. (need to be done.)

Please set the environment as 

Pytorch == 1.10.1+cu102
numpy == 1.19.5
python == 3.6.9 64- bit.

The experiments are on a single NVIDIA Tesla V100 32G.

Please download all the datasets under the path 'data', from the baseline work ODIN in its public code: https://github.com/facebookresearch/odin. To start the two-phase training, you need run:

CUDA_VISIBLE_DEVICES=0 python3 train.py --cfg ./configs/ood/cifar100.yaml

and move the saved trained model under the path 'saved_model/backbone/'. Next run the code:

CUDA_VISIBLE_DEVICES=0 python3 train.py --cfg ./configs/ood/cifar100_ood.yaml 

Test result will be automatically printed while training. 
