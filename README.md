# About this code
This is the official implementation of CVPR2022 paper 
'Rethinking Reconstruction Autoencoder-Based Out-of-Distribution Detection'. The original benchmark results in paper 
were produced in Tensorflow 1.14.0, and this codebase is a pytorch version of 
re-implementing it. Thus, there might be minor differences regarding the 
original results. Specifically, 

* In this code, we do not partion the validation set for the 'scale' and 'location' fitting of Gaussian CDFs.
Instead we just do it on the test set, which yields similar performance with that in the paper.
* Some training configs like learning scheduler are different from that in paper.
* The results on iSUN, tinyImageNet-resize and largeSUN-resize are slightly better than that in paper, while those on largeSUN-crop and tinyImageNet-crop are not. The reason-why is still to be figured out. 


## Results
The results on WRN-28-10 with CIFAR-100 is (FPR@95%TPR, detection error, AUROC and AUPR)

![image](https://github.com/SDret/Pytorch-implementation-for-Rethinking-Reconstruction-Autoencoder-Based-Out-of-Distribution-Detection/blob/main/results.png)


If the Tensorflow original implementation of this work is desired, feel free to contact us with ybzhou@buaa.edu.cn
## Environment
Please set the environment as

Pytorch == 1.10.1+cu102 

numpy == 1.19.5 python == 3.6.9 64- bit.


## Start training

Please mkdir the folder 'data', and download all the datasets into it from the baseline work ODIN: https://github.com/facebookresearch/odin. Training logs and trained model parameters would be saved under the path 'exp_result'.

To start the two-phase training, you need run:

```
CUDA_VISIBLE_DEVICES=0 python3 train.py --cfg ./configs/ood/cifar100.yaml
```
and move the saved trained model under the path 'saved_model/backbone/$BACKBONETYPE$'. Next, apply this path in the config file 'cifar100_ood.yaml' and run the code:

```
CUDA_VISIBLE_DEVICES=0 python3 train.py --cfg ./configs/ood/cifar100_ood.yaml
```
Benchmark results will be automatically printed while training.

