from nonlocal_net import *
from symbol_resnet import *


def get_symbol(cfg):
    """Return non local ResNet symbol of cifar10 and imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_stage = cfg.network.num_stages
    filter_list = cfg.network.filter_list
    units = cfg.network.units
    data_type = cfg.dataset.data_type
    num_class = cfg.dataset.num_classes
    bn_mom = cfg.train.bn_mom
    workspace = cfg.train.workspace
    resample = cfg.nonlocal.resample
    bottle_neck=True
    memonger= cfg.memonger
    
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type in ['cifar10', 'cifar100']:
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    elif data_type in ['imagenet', 'tiny-imagenet']:
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    else:
         raise ValueError("do not support {} yet".format(data_type))

    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
                
            ## add non_local_block right before last residual block of a stage
            if (i == 1 or i == 2) and (j == units[i]-2):
                print("Filters : ", filter_list[i+1], "Adding Non Local Blocks, at Stage ", i+1)
                body = non_local_block(body, filter_list[i+1], mode='embedded', resample=resample, ith=(i*100+j+1))
        
    # print("Filters : ", filter_list[i+1], "Adding Non Local Blocks")
    # body = non_local_block(body, filter_list[i+1], mode='embedded', resample=resample, ith=(i))
    
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_class, name='fc1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
