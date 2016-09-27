"""Wrapping of certain ops for positional arguments.

Mainly because NNVM accepts kwargs for some additional arguments,
while TF sometimes support positional ops.
"""
from __future__ import absolute_import as _abs
from nnvm import symbol
from nnvm import _symbol_internal


def argmax(x, axis):
    return _symbol_internal._argmax(x, reduction_indices=[axis])
