// Copyright (c) 2016 by Contributors
// implementation of common nn operators
#include <tinyflow/base.h>
#include <nnvm/op_attr_types.h>
#include <utility>
#include "./op_util.h"

namespace tinyflow {

using namespace nnvm;

const FLuaCompute kLuaNOP = "function(x, y, kwarg) return function() end end";

NNVM_REGISTER_OP(placeholder)
.describe("placeholder op")
.set_num_inputs(0);

template<typename Attr>
inline bool EmptyAttr(const NodeAttrs& attrs,
                      std::vector<Attr> *ishape,
                      std::vector<Attr> *oshape) {
  oshape->at(0) = Attr{0}; return true;
}

NNVM_REGISTER_OP(_nop)
.describe("no operation")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", EmptyAttr<TShape>)
.set_attr<FInferType>("FInferType", EmptyAttr<int>);


NNVM_REGISTER_OP(assign)
.describe("assign second to the first")
.set_num_inputs(2)
.set_attr<FMutateInputs>("FMutateInputs", [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>{0};
  })
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn1Out0);

// special no gradient op to report error when take
// gradient wrt non-differentiable inputs
NNVM_REGISTER_OP(_no_gradient)
.describe("Special op indicating no gradient")
.set_num_inputs(0);

}  // namespace tinyflow
