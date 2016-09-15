from __future__ import absolute_import
import os
import sys
if sys.version_info[0] == 3:
    import builtins as __builtin__
else:
    import __builtin__

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

if hasattr(__builtin__, "NNVM_BASE_PATH"):
    assert __builtin__.NNVM_BASE_PATH == curr_path
else:
    __builtin__.NNVM_BASE_PATH = curr_path

if hasattr(__builtin__, "NNVM_LIBRARY_NAME"):
    assert __builtin__.NNVM_LIBRARY_NAME == curr_path
else:
    __builtin__.NNVM_LIBRARY_NAME = "libtinyflow"


import ctypes as _ctypes
from nnvm.name import NameManager
from nnvm import symbol
from nnvm._base import c_str, check_call, _LIB

# data type table
float32 = 0

def placeholder(dtype, shape=None, name=None):
    v = symbol.placeholder(name=name, dtype=dtype)
    return v

def Variable(init, name=None):
    if not isinstance(init, symbol.Symbol):
        raise TypeError("Expect initialization expression to be Symbol")
    name = NameManager.current.get(name, 'variable')
    v = symbol.Variable(name)
    return v
