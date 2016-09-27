/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api.h
 * \brief C API to tiny flow
 */
#ifndef TINYFLOW_C_API_H_
#define TINYFLOW_C_API_H_

#include <nnvm/c_api.h>

typedef void* SessionHandle;

NNVM_DLL int NNSessionCreate(SessionHandle* handle, const char* option);

NNVM_DLL int NNSessionClose(SessionHandle handle);

NNVM_DLL int NNSessionRun(SessionHandle handle,
                          SymbolHandle graph,
                          nn_uint num_feed,
                          const SymbolHandle* feed_placeholders,
                          const float** feed_dptr,
                          const nn_uint* feed_dtype,
                          const nn_uint* feed_shape_csr_ptr,
                          const nn_uint* feed_shape_data,
                          nn_uint* num_out,
                          const float*** out_dptr,
                          const nn_uint** out_dtype,
                          const nn_uint **out_shape_ndim,
                          const nn_uint ***out_shape_data);

#endif  // TINYFLOW_C_API_H_
