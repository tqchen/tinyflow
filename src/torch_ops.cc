// Copyright (c) 2016 by Contributors
#include <tinyflow/base.h>

namespace tinyflow {

NNVM_REGISTER_OP(__add_symbol__)
.set_attr<FLuaComputeCode>(
    "FLuaComputeCode", R"(
function(x, y)
  torch.add(y[1], x[1], x[2])
end
)");

// set to nop
NNVM_REGISTER_OP(placeholder)
.set_attr<FLuaComputeCode>(
    "FLuaComputeCode", R"(
function(x, y)
end
)");

NNVM_REGISTER_OP(__mul_symbol__)
.set_attr<FLuaComputeCode>(
    "FLuaComputeCode", R"(
function(x, y)
  torch.cmul(y[1], x[1], x[2])
end
)");


NNVM_REGISTER_OP(__mul_scalar__)
.set_attr<FLuaComputeCode>(
    "FLuaComputeCode", R"(
function(x, y)
  torch.mul(y[1], x[1], x[2])
end
)");

NNVM_REGISTER_OP(zeros)
.set_attr<FLuaComputeCode>(
    "FLuaComputeCode", R"(
function(x, y)
  y[1]:fill(0)
end
)");

NNVM_REGISTER_OP(assign)
.set_attr<FLuaComputeCode>(
    "FLuaComputeCode", R"(
function(x, y)
  y[1]:copy(x)
end
)");

}  // namespace tinyflow
