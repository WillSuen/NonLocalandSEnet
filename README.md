# NonLocalNetwork and Sequeeze-Excitation Network

### Paper:
1. Non-local Neural Networks (https://arxiv.org/abs/1711.07971)
2. Squeeze-and-Excitation Networks (https://arxiv.org/abs/1709.01507)

#### Train nonlocal with cifar100
`python train_nonlocal.py --cfg cfgs/nonlocal_cifar100.yaml`


#### Train sequeeze-excitation with tiny-imagenet
`python train_seresnet.py --cfg cfgs/seresnet_tinyimagenet.yaml`


### Reference
[1] [SENet.mxnet](https://github.com/bruinxiong/SENet.mxnet)

[2] [keras-non-local-nets](https://github.com/titu1994/keras-non-local-nets)