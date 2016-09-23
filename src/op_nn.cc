// Copyright (c) 2016 by Contributors
// implementation of common nn operators
#include <tinyflow/base.h>
#include <utility>
#include "./op_util.h"

namespace tinyflow {

NNVM_REGISTER_OP(softmax)
.describe("Softmax operation")
.set_num_inputs(1);

NNVM_REGISTER_OP(bias_add)
.describe("Add bias to data")
.set_num_inputs(2);

}  // namespace tinyflow
