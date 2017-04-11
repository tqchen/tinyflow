// Copyright (c) 2016 by Contributors
// implementation of common tensor operators
#include <tinyflow/base.h>

namespace tinyflow {

NNVM_REGISTER_OP(zeros)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      y[1]:fill(0)
    end
  end
)");


NNVM_REGISTER_OP(zeros_like)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      y[1]:fill(0)
    end
  end
)");


NNVM_REGISTER_OP(ones)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      y[1]:fill(1)
    end
  end
)");


NNVM_REGISTER_OP(ones_like)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      y[1]:fill(1)
    end
  end
)");


NNVM_REGISTER_OP(normal)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      local scale = 1
      if kwarg.stdev ~= nil then
        scale = tonumber(kwarg.stdev)
      end
      y[1]:copy(torch.randn(y[1]:size()) * scale)
    end
  end
)");


NNVM_REGISTER_OP(equal)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      y[1]:copy(torch.eq(x[1], x[2]))
    end
  end
)");


NNVM_REGISTER_OP(__ewise_sum__)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      y[1]:copy(x[1])
      for i = 2, #x do
        torch.add(y[1], y[1], x[i])
      end
    end
  end
)");


NNVM_REGISTER_OP(__add_symbol__)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      torch.add(y[1], x[1], x[2])
    end
  end
)");


NNVM_REGISTER_OP(__add_scalar__)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    local scalar = tonumber(kwarg.scalar)
    return function()
      torch.add(y[1], x[1], scalar)
    end
  end
)");


NNVM_REGISTER_OP(__sub_symbol__)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      torch.add(y[1], x[1], -x[2])
    end
  end
)");


NNVM_REGISTER_OP(__sub_scalar__)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    local scalar = tonumber(kwarg.scalar)
    return function()
      torch.add(y[1], x[1], -scalar)
    end
  end
)");


NNVM_REGISTER_OP(__rsub_scalar__)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    local scalar = tonumber(kwarg.scalar)
    return function()
      torch.add(y[1], -x[1], scalar)
    end
  end
)");


NNVM_REGISTER_OP(__mul_symbol__)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      if x[1]:dim() == 1 and x[1]:size(1) == 1 then
        local scalar = x[1][1]
        torch.mul(y[1], x[2], scalar)
        return
      end
      if x[2]:dim() == 1 and x[2]:size(1) == 1 then
        local scalar = x[2][1]
        torch.mul(y[1], x[1], scalar)
        return
      end
      torch.cmul(y[1], x[1], x[2])
    end
  end
)");


NNVM_REGISTER_OP(__mul_scalar__)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    local scalar = tonumber(kwarg.scalar)
    return function()
      torch.mul(y[1], x[1], scalar)
    end
  end
)");


NNVM_REGISTER_OP(__div_symbol__)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      torch.cdiv(y[1], x[1], x[2])
    end
  end
)");


NNVM_REGISTER_OP(__div_scalar__)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    local scalar = tonumber(kwarg.scalar)
    return function()
      torch.div(y[1], x[1], scalar)
    end
  end
)");


NNVM_REGISTER_OP(exp)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      torch.exp(y[1], x[1])
    end
  end
)");


NNVM_REGISTER_OP(log)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      torch.log(y[1], x[1])
    end
  end
)");


NNVM_REGISTER_OP(sqrt)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      torch.sqrt(y[1], x[1])
    end
  end
)");


NNVM_REGISTER_OP(__pow_symbol__)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      torch.cpow(y[1], x[1], x[2])
    end
  end
)");


NNVM_REGISTER_OP(__rpow_scalar__)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    local scalar = tonumber(kwarg.scalar)
    return function()
      torch.pow(y[1], scalar, x[1])
    end
  end
)");


NNVM_REGISTER_OP(matmul)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      torch.mm(y[1], x[1], x[2])
    end
  end
)");


// simply register a bulk op for backward
NNVM_REGISTER_OP(_matmul_backward)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    local gradOutput = x[1]
    local lhs = x[2]
    local rhs = x[3]
    local gradLhs = y[1]
    local gradRhs = y[2]
    return function()
      torch.mm(gradRhs, lhs:t(), gradOutput)
      torch.mm(gradLhs, gradOutput, rhs:t())
    end
  end
)");


NNVM_REGISTER_OP(reduce_sum)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    local rhs = x[1]
    local lhs = y[1]
    if kwarg.reduction_indices == nil then
      rhs = rhs:view(rhs:nElement())
      return function()
        torch.sum(lhs, rhs, 1)
      end
    else
      local axis = nn_parse_tuple(kwarg.reduction_indices)
      table.sort(axis)
      local k = #axis
      return function()
        for i = 1, (k - 1) do
          rhs = torch.sum(rhs, axis[k - i + 1] + 1)
        end
        torch.sum(lhs, rhs, axis[1] + 1)
      end
    end
  end
)");


NNVM_REGISTER_OP(reduce_mean)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    local rhs = x[1]
    local lhs = y[1]
    if kwarg.reduction_indices == nil then
      rhs = rhs:view(rhs:nElement())
      return function()
        torch.mean(lhs, rhs, 1)
      end
    else
      local axis = nn_parse_tuple(kwarg.reduction_indices)
      table.sort(axis)
      local k = #axis
      return function()
        for i = 1, (k - 1) do
          rhs = torch.mean(rhs, axis[k - i + 1] + 1)
        end
        torch.mean(lhs, rhs, axis[1] + 1)
      end
    end
  end
)");


NNVM_REGISTER_OP(_reduce_sum_backward)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    local rhs = x[1]
    local lhs = y[1]
    if kwarg.reduction_indices == nil then
      lhs = lhs:view(lhs:nElement())
      rhs = rhs:expandAs(lhs)
    else
      local axis = nn_parse_tuple(kwarg.reduction_indices)
      local vshape = lhs:size()
      for i = 1, #axis do
        vshape[axis[i] + 1] = 1
      end
      rhs = rhs:view(vshape):expandAs(lhs)
    end
    return function()
      lhs:copy(rhs)
    end
  end
)");


NNVM_REGISTER_OP(_reduce_mean_backward)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    local rhs = x[1]
    local lhs = y[1]
    local scale = 1
    if kwarg.reduction_indices == nil then
      lhs = lhs:view(lhs:nElement())
      rhs = rhs:expandAs(lhs)
      scale = lhs:nElement()
    else
      local axis = nn_parse_tuple(kwarg.reduction_indices)
      local vshape = lhs:size()
      for i = 1, #axis do
        scale = scale * vshape[axis[i] + 1]
        vshape[axis[i] + 1] = 1
      end
      rhs = rhs:view(vshape):expandAs(lhs)
    end
    return function()
      torch.div(lhs, rhs, scale)
    end
  end
)");


NNVM_REGISTER_OP(_argmax)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    local rhs = x[1]
    local lhs = y[1]
    local axis = nn_parse_tuple(kwarg.reduction_indices)
    return function()
      local mx, ind = torch.max(rhs, axis[1] + 1)
      torch.add(lhs, ind:typeAs(lhs), -1)
    end
  end
)");

} // namespace tinyflow
