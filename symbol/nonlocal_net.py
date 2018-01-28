import mxnet as mx
from operator_py.RmSelfAtten import *


def non_local_block(insym, num_filter, mode='Embedded Gaussian', resample=True, ith=0):
    """Return nonlocal neural network block
    Parameters
    ----------
    insym : mxnet symbol
        Input symbol
    num_filter : int
        Number of input channels
    mode : str
        `mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`
    """
    # default mode is Embedded Gaussian
    new_filter = num_filter / 2 if num_filter >= 1024 else num_filter
    indata1 = mx.sym.Convolution(insym, kernel=(1, 1), stride=(1, 1), num_filter=new_filter,
                                 no_bias=True, name='nonlocal_conv%d1' % ith)
    indata2 = mx.sym.Convolution(insym, kernel=(1, 1), stride=(1, 1), num_filter=new_filter,
                                 no_bias=True, name='nonlocal_conv%d2' % ith)

    # data size: batch_size x (num_filter / 2) x HW
    indata1 = mx.sym.reshape(indata1, shape=(0, 0, -1))
    indata2 = mx.sym.reshape(indata2, shape=(0, 0, -1))

    # f size: batch_size x HW x HW
    f = mx.sym.batch_dot(lhs=indata1, rhs=indata2, transpose_a=True, name='nonlocal_dot%d1' % ith)

    # add softmax layer
    f = mx.sym.softmax(f, axis=2)

    indata3 = mx.sym.Convolution(insym, kernel=(1, 1), stride=(1, 1), num_filter=new_filter,
                                 no_bias=True, name='nonlocal_conv3%d' % ith)
    # g size: batch_size x (num_filter / 2) x HW
    g = mx.sym.reshape(indata3, shape=(0, 0, -1))

    y = mx.sym.batch_dot(lhs=f, rhs=g, transpose_b=True, name='nonlocal_dot%d2' % ith)
    y = mx.sym.reshape_like(lhs=mx.sym.transpose(y, axes=(0, 2, 1)), rhs=indata3)
    # y = mx.sym.reshape_like(lhs=y, rhs=indata3)
    y = mx.sym.Convolution(y, kernel=(1, 1), stride=(1, 1), num_filter=num_filter,
                           no_bias=True, name='nonlocal_conv%d4' % ith)
    outsym = insym + y
    return outsym
