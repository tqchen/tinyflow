// Copyright (c) 2016 by Contributors
// implementation of common nn operators
#include <tinyflow/base.h>
#include <utility>
#include "../op_util.h"

namespace tinyflow {

NNVM_REGISTER_OP(softmax)
.set_attr<FLuaCreateNNModule>(
  "FLuaCreateNNModule", R"(
  function(ishape, kwarg)
    return nn.SoftMax()
  end
)");


NNVM_REGISTER_OP(tanh)
.set_attr<FLuaCreateNNModule>(
  "FLuaCreateNNModule", R"(
  function(ishape, kwarg)
    return nn.Tanh()
  end
)");


NNVM_REGISTER_OP(relu)
.set_attr<FLuaCreateNNModule>(
  "FLuaCreateNNModule", R"(
  function(ishape, kwarg)
    return nn.ReLU()
  end
)");


NNVM_REGISTER_OP(linear)
.set_attr<FLuaCreateNNModule>(
  "FLuaCreateNNModule", R"(
  function(ishape, kwarg)
    local wshape = ishape[2]
    local m = nn.Linear(wshape[2], wshape[1])
    if #ishape == 2 then
      m = m:noBias()
    end
    return m
  end
)");


NNVM_REGISTER_OP(pad)
.set_attr<FLuaCreateNNModule>(
  "FLuaCreateNNModule", R"(
  function(ishape, kwarg)
    local dim = tonumber(kwarg.dim) + 1
    local pad = tonumber(kwarg.pad)
    local m = nn.Padding(dim, pad)
    return m
  end
)");


NNVM_REGISTER_OP(conv2d)
.set_attr<FLuaCreateNNModule>(
  "FLuaCreateNNModule", R"(
  function(ishape, kwarg)
    local dshape = ishape[2]
    local fshape = ishape[2]
    local outPlane = fshape[1]
    local inPlane = fshape[2]
    local kH = fshape[3]
    local kW = fshape[4]
    local inH = dshape[3]
    local inW = dshape[4]
    local stride = nn_parse_tuple(kwarg.strides, {1,1,1,1})
    local dH = stride[2]
    local dW = stride[3]
    local padH = 0
    local padW = 0

    assert(kwarg.data_format == 'NCHW')
    if kwarg.padding == 'SAME' then
      padW = math.floor((kW - 1) / 2)
      padH = math.floor((kH - 1) / 2)
    end
    local m = nn.SpatialConvolution(
      inPlane, outPlane,
      kW, kH, dW, dH, padW, padH)
    if #ishape == 2 then
      m = m:noBias()
    end
    return m
  end
)");


NNVM_REGISTER_OP(max_pool)
.set_attr<FLuaCreateNNModule>(
  "FLuaCreateNNModule", R"(
  function(ishape, kwarg)
    local ksize = nn_parse_tuple(kwarg.ksize)
    local stride = nn_parse_tuple(kwarg.strides, {1,1,1,1})
    local kH = ksize[2]
    local kW = ksize[3]
    local dH = stride[2]
    local dW = stride[3]
    local padH = 0
    local padW = 0
    assert(kwarg.data_format == 'NCHW')
    if kwarg.padding == 'SAME' then
      padW = math.floor((kW - 1) / 2)
      padH = math.floor((kH - 1) / 2)
    end
    return nn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
  end
)");


NNVM_REGISTER_OP(avg_pool)
.set_attr<FLuaCreateNNModule>(
  "FLuaCreateNNModule", R"(
  function(ishape, kwarg)
    local ksize = nn_parse_tuple(kwarg.ksize)
    local stride = nn_parse_tuple(kwarg.strides, {1,1,1,1})
    local kH = ksize[2]
    local kW = ksize[3]
    local dH = stride[2]
    local dW = stride[3]
    local padH = 0
    local padW = 0
    assert(kwarg.data_format == 'NCHW')
    if kwarg.padding == 'SAME' then
      padW = math.floor((kW - 1) / 2)
      padH = math.floor((kH - 1) / 2)
    end
    local m = nn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH)
    return m
  end
)");


NNVM_REGISTER_OP(batch_normalization)
.set_attr<FLuaCreateNNModule>(
  "FLuaCreateNNModule", R"(
  function(ishape, kwarg)
    local n = ishape[1][2]
    return nn.SpatialBatchNormalization(n)
  end
)");


NNVM_REGISTER_OP(mean_sparse_softmax_cross_entropy_with_logits)
.set_attr<FLuaCreateNNModule>(
  "FLuaCreateNNModule", R"(
  function(ishape, kwarg)
    return nn_zero_index_target_criterion(
      nn.CrossEntropyCriterion())
  end
)");


const char* LuaReshape = R"(
function(x, y, kwarg)
  if x[1]:storage() == y[1]:storage() then
    return function() end
  else
    return function() y[1]:copy(x[1]:resizeAs(y[1])) end
  end
end
)";


NNVM_REGISTER_OP(flatten_layer)
.set_attr<FLuaCompute>("FLuaCompute", LuaReshape);


NNVM_REGISTER_OP(_flatten_backward)
.set_attr<FLuaCompute>("FLuaCompute", LuaReshape);

}  // namespace tinyflow
