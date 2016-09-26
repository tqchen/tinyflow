// Copyright (c) 2016 by Contributors
// implementation of common nn operators
#include <tinyflow/base.h>
#include <utility>
#include "./op_util.h"

namespace tinyflow {

using namespace nnvm;

const FLuaCompute kLuaNOP = "function(x, y) end";

NNVM_REGISTER_OP(placeholder)
.describe("placeholder op")
.set_num_inputs(0)
.set_attr<FLuaCompute>("FLuaCompute", kLuaNOP);

NNVM_REGISTER_OP(assign)
.describe("assign second to the first")
.set_num_inputs(2)
.set_attr<FMutateInputs>("FMutateInputs", [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>{0};
  })
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn1Out0)
.set_attr<FLuaCompute>(
    "FLuaCompute", R"(
function(x, y)
  x[1]:copy(x[2])
  -- normally inplace optimization prevent this
  if y[1]:storage() ~= x[2]:storage() then
    y[1]:copy(x[2])
  end
end
)");

// special no gradient op to report error when take
// gradient wrt non-differentiable inputs
NNVM_REGISTER_OP(_no_gradient)
.describe("Special op indicating no gradient")
.set_num_inputs(0);

// special backward op indicating backward of nn module
NNVM_REGISTER_OP(_backward)
.describe("backward operator of NN module")
.set_num_outputs([] (const NodeAttrs& attrs) {
  const BackwardParam& param = dmlc::get<BackwardParam>(attrs.parsed);
  return param.forward_readonly_inputs;
  })
.set_num_inputs([] (const NodeAttrs& attrs) {
  const BackwardParam& param = dmlc::get<BackwardParam>(attrs.parsed);
  uint32_t n = param.num_states + 1;
  if (param.need_inputs) n += param.forward_readonly_inputs;
  if (param.need_outputs) n += 1;
  return n;
  })
.set_attr<nnvm::FBackwardOutToInIndex>("FBackwardOutToInIndex", [](const NodeAttrs& attrs) {
  const BackwardParam& param = dmlc::get<BackwardParam>(attrs.parsed);
  std::vector<uint32_t> vec;
  for (uint32_t i = 0; i < param.forward_readonly_inputs; ++i) {
    vec.push_back(i);
  }
  return vec;
  })
.set_attr<nnvm::FBackwardInGradIndex>("FBackwardInGradIndex", [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>{0};
  });

}  // namespace tinyflow
