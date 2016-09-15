// Copyright (c) 2016 by Contributors
#include <tinyflow/base.h>
#include <nnvm/base.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/node.h>
#include <nnvm/graph_attr_types.h>
#include <utility>

namespace tinyflow {

using namespace nnvm;

// simply return the shape as same
inline bool SameShape(const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
  if (ishape->size() == 0 || (*ishape)[0].ndim() == 0) return false;
  for (TShape& pshape : *oshape) {
    pshape = (*ishape)[0];
  }
  for (TShape& pshape : *ishape) {
    pshape = (*ishape)[0];
  }
  return true;
}

inline std::vector<std::pair<int, int> > InplaceIn0Out0(const NodeAttrs& attrs) {
  return {{0, 0}};
}

NNVM_REGISTER_OP(placeholder)
.describe("placeholder op")
.set_num_inputs(0);

NNVM_REGISTER_OP(zeros)
.describe("zeros")
.set_num_inputs(0);


NNVM_REGISTER_OP(assign)
.describe("assign second to the first")
.set_num_inputs(2)
.set_attr<FMutateInputs>("FMutateInputs", [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>{0};
  });

NNVM_REGISTER_OP(__add_symbol__)
.describe("add two data together")
.set_num_inputs(2)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{ograds[0], ograds[0]};
    });


NNVM_REGISTER_OP(__mul_symbol__)
.describe("add two data together")
.set_num_inputs(2)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0);

NNVM_REGISTER_OP(__mul_scalar__)
.describe("Multiply symbol with scalar")
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0);

NNVM_REGISTER_OP(matmul)
.describe("Matrix multiplication")
.set_num_inputs(2);

NNVM_REGISTER_OP(softmax)
.describe("Softmax operation")
.set_num_inputs(1);

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
