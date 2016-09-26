from __future__ import absolute_import as _abs
import ctypes as _ctypes
import numpy as np
from nnvm import symbol
from nnvm._base import c_str, check_call, _LIB, c_array, nn_uint

SessionHandle = _ctypes.c_void_p
nn_float = _ctypes.c_float

class Session(object):
    def __init__(self):
        handle = SessionHandle()
        check_call(_LIB.NNSessionCreate(_ctypes.byref(handle)))
        self.handle = handle

    def __del__(self):
        check_call(_LIB.NNSessionClose(self.handle))

    def run(self, fetch, feed_dict=None):
        feed_dict = feed_dict if feed_dict else {}
        num_feed = len(feed_dict)
        feed_placeholders = []
        feed_dptr = []
        feed_dtype = []
        feed_shape_csr_ptr = [0]
        feed_shape_data = []

        for k, v in feed_dict.items():
            assert isinstance(k, symbol.Symbol)
            assert isinstance(v, np.ndarray)
            feed_placeholders.append(k.handle)
            # only convert to float32 for now
            source_array = np.ascontiguousarray(v, dtype=np.float32)
            feed_dptr.append(source_array.ctypes.data_as(_ctypes.c_void_p))
            feed_dtype.append(0)
            feed_shape_data.extend(source_array.shape)
            feed_shape_csr_ptr.append(len(feed_shape_data))
        out_size = nn_uint()
        out_dptr = _ctypes.POINTER(_ctypes.POINTER(nn_float))()
        out_shape_ndim = _ctypes.POINTER(nn_uint)()
        out_shape_data = _ctypes.POINTER(_ctypes.POINTER(nn_uint))()

        check_call(_LIB.NNSessionRun(
            self.handle, fetch.handle, nn_uint(num_feed),
            c_array(_ctypes.c_void_p, feed_placeholders),
            c_array(_ctypes.c_void_p, feed_dptr),
            c_array(nn_uint, feed_dtype),
            c_array(nn_uint, feed_shape_csr_ptr),
            c_array(nn_uint, feed_shape_data),
            _ctypes.byref(out_size),
            _ctypes.byref(out_dptr),
            _ctypes.byref(out_shape_ndim),
            _ctypes.byref(out_shape_data)))
        ret = []
        for i in range(out_size.value):
            shape = tuple(out_shape_data[i][:out_shape_ndim[i]])
            size = 1
            for s in shape:
                size *= s

            if size != 0 and shape:
                cptr = out_dptr[i]
                dbuffer = (nn_float * size).from_address(_ctypes.addressof(cptr.contents))
                nd = np.frombuffer(dbuffer, dtype=np.float32).reshape(shape).copy()
                ret.append(nd)
        return ret[0] if len(ret) == 1 else ret
