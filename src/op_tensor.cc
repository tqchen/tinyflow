// Copyright (c) 2016 by Contributors
// implementation of common tensor operators
#include <tinyflow/base.h>
#include <dmlc/parameter.h>
#include <utility>
#include "./op_util.h"

namespace tinyflow {

// shape parameter for zeros, ones
struct ShapeParam : public dmlc::Parameter<ShapeParam> {
  TShape shape;
  DMLC_DECLARE_PARAMETER(ShapeParam) {
    DMLC_DECLARE_FIELD(shape);
  }
};
DMLC_REGISTER_PARAMETER(ShapeParam);

// shape given the ShapeParam


using namespace nnvm;

const FLuaComputeCode kLuaNOP = "function(x, y) end";

NNVM_REGISTER_OP(placeholder)
.describe("placeholder op")
.set_num_inputs(0)
.set_attr<FLuaComputeCode>("FLuaComputeCode", kLuaNOP);


NNVM_REGISTER_OP(assign)
.describe("assign second to the first")
.set_num_inputs(2)
.set_attr<FMutateInputs>("FMutateInputs", [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>{0};
  })
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn1Out0)
.set_attr<FLuaComputeCode>(
    "FLuaComputeCode", R"(
function(x, y)
  x[1]:copy(x[2])
  -- normally inplace optimization prevent this
  if y[1]:storage() ~= x[2]:storage() then
    y[1]:copy(x[2])
  end
end
)");

inline bool ZeroShape(const NodeAttrs& attrs,
                       std::vector<TShape> *ishape,
                       std::vector<TShape> *oshape) {
  oshape->at(0) = dmlc::get<ShapeParam>(attrs.parsed).shape;
  return true;
}

NNVM_REGISTER_OP(zeros)
.describe("zeros")
.set_num_inputs(0)
.set_attr_parser(ParamParser<ShapeParam>)
.set_attr<FInferShape>("FInferShape", ZeroShape)
.set_attr<FLuaComputeCode>(
    "FLuaComputeCode", R"(
function(x, y)
  y[1]:fill(0)
end
)");


NNVM_REGISTER_OP(__add_symbol__)
.describe("add two data together")
.set_num_inputs(2)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{ograds[0], ograds[0]};
    })
.set_attr<FLuaComputeCode>(
    "FLuaComputeCode", R"(
function(x, y)
  torch.add(y[1], x[1], x[2])
end
)");


NNVM_REGISTER_OP(__mul_symbol__)
.describe("add two data together")
.set_num_inputs(2)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FLuaComputeCode>(
    "FLuaComputeCode", R"(
function(x, y)
  torch.cmul(y[1], x[1], x[2])
end
)");


NNVM_REGISTER_OP(__mul_scalar__)
.describe("Multiply symbol with scalar")
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0);


NNVM_REGISTER_OP(matmul)
.describe("Matrix multiplication")
.set_num_inputs(2);


NNVM_REGISTER_OP(reduce_mean)
.describe("reduce mean")
.set_num_inputs(1);


NNVM_REGISTER_OP(reduce_sum)
.describe("reduce mean")
.set_num_inputs(1);


NNVM_REGISTER_OP(log)
.describe("reduce mean")
.set_num_inputs(1);
}  // namespace tinyflow
