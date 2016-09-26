// Copyright (c) 2016 by Contributors
// implementation of common nn operators
#include <tinyflow/base.h>
#include <utility>
#include "./op_util.h"

namespace tinyflow {

using namespace nnvm;

// create a backward node
inline std::vector<NodeEntry> MakeBackwardNode(
    const NodePtr& n,
    const std::vector<NodeEntry>& ograds) {
  static auto& backward_need_inputs = Op::GetAttr<bool>("TBackwardNeedInputs");
  static auto& backward_need_outputs = Op::GetAttr<bool>("TBackwardNeedOutputs");
  nnvm::NodePtr p = nnvm::Node::Create();
  p->attrs.op = nnvm::Op::Get("_backward");
  p->attrs.name = n->attrs.name + "_backward";

  BackwardParam param;
  param.forward_readonly_inputs = static_cast<uint32_t>(n->inputs.size());
  param.need_inputs = backward_need_inputs[n->op()];
  param.need_outputs = backward_need_outputs[n->op()];
  CHECK_EQ(ograds.size(), 1);
  CHECK_EQ(param.forward_readonly_inputs + param.num_states,
           static_cast<uint32_t>(n->inputs.size()));

  p->attrs.parsed = param;
  p->control_deps.emplace_back(n);
  // layout [output_grad, inputs, states, output]
  p->inputs.push_back(ograds[0]);
  if (param.need_inputs) {
    for (index_t i = 0; i < param.forward_readonly_inputs; ++i) {
      p->inputs.push_back(n->inputs[i]);
    }
  }
  for (index_t i = 0; i < param.num_states; ++i) {
    p->inputs.push_back(n->inputs[i + param.forward_readonly_inputs]);
  }
  if (param.need_outputs) {
    for (uint32_t i = 0; i < n->num_outputs(); ++i) {
      p->inputs.emplace_back(nnvm::NodeEntry{n, i, 0});
    }
  }
  std::vector<nnvm::NodeEntry> ret;
  for (index_t i = 0; i < param.forward_readonly_inputs; ++i) {
    ret.emplace_back(nnvm::NodeEntry{p, i, 0});
  }
  if (param.num_states != 0) {
    nnvm::NodePtr np = nnvm::Node::Create();
    np->attrs.op = nnvm::Op::Get("_no_gradient");
    for (index_t i = 0; i < param.num_states; ++i) {
      ret.emplace_back(nnvm::NodeEntry{np, i, 0});
    }
  }
  return ret;
}

// common attributes for nn module.
NNVM_REGISTER_OP_GROUP(nn_module)
.set_attr<FGradient>("FGradient", MakeBackwardNode)
.set_attr<bool>("TBackwardNeedInputs", true)
.set_attr<bool>("TBackwardNeedOutputs", true);


NNVM_REGISTER_OP(softmax)
.describe("Softmax operation")
.set_num_inputs(1)
.include("nn_module")
.set_attr<FLuaCreateNNModule>(
    "FLuaCreateNNModule", R"(
function(ishape, oshape, kwarg)
  return nn.SoftMax()
end
)")
.set_attr<FInferShape>("FInferShape", SameShape);

}  // namespace tinyflow
