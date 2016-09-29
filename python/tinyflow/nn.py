from nnvm.symbol import *
from nnvm import symbol as _sym

def conv2d(data, weight=None,
           strides=[1, 1, 1, 1],
           padding='VALID',
           data_format='NCHW',
           **kwargs):
    kwargs = kwargs.copy()
    kwargs['data'] = data
    if weight:
        kwargs['weight'] = weight
    return _sym.conv2d(strides=strides, padding=padding, data_format=data_format, **kwargs)

def max_pool(data,
             strides=[1, 1, 1, 1],
             padding='VALID',
             data_format='NCHW', **kwargs):
    return _sym.max_pool(data, strides=strides, padding=padding,
                         data_format=data_format, **kwargs)
