// Copyright (c) 2016 by Contributors
// implementation of common nn operators
#include <tinyflow/base.h>

namespace tinyflow {

const FLuaCompute kLuaNOP = "function(x, y, kwarg) return function() end end";

NNVM_REGISTER_OP(placeholder)
.set_attr<FLuaCompute>("FLuaCompute", kLuaNOP);

NNVM_REGISTER_OP(_nop)
.set_attr<FLuaCompute>("FLuaCompute", kLuaNOP);

NNVM_REGISTER_OP(assign)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      x[1]:copy(x[2])
      -- normally inplace optimization prevent this
      if y[1]:storage() ~= x[2]:storage() then
        y[1]:copy(x[2])
      end
    end
  end
)");

}  // namespace tinyflow
