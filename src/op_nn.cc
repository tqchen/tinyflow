// Copyright (c) 2016 by Contributors
// implementation of common nn operators
#include <tinyflow/base.h>
#include <utility>
#include "./op_util.h"

namespace tinyflow {

using namespace nnvm;

// create a backward node
inline std::vector<NodeEntry> MakeNNBackwardNode(
    const NodePtr& n,
    const std::vector<NodeEntry>& ograds) {
  static auto& backward_need_inputs = Op::GetAttr<bool>("TBackwardNeedInputs");
  static auto& backward_need_outputs = Op::GetAttr<bool>("TBackwardNeedOutputs");
  static auto& backward_num_nograd = Op::GetAttr<int>("TBackwardNumNoGradInputs");
  nnvm::NodePtr p = nnvm::Node::Create();
  p->attrs.op = nnvm::Op::Get("_backward");
  p->attrs.name = n->attrs.name + "_backward";

  NNBackwardParam param;
  param.forward_readonly_inputs = static_cast<uint32_t>(n->inputs.size());
  param.need_inputs = backward_need_inputs[n->op()];
  param.need_outputs = backward_need_outputs[n->op()];
  param.num_no_grad_inputs = backward_num_nograd.get(n->op(), 0);
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
  if (param.num_states != 0 || param.num_no_grad_inputs != 0) {
    nnvm::NodePtr np = nnvm::Node::Create();
    np->attrs.op = nnvm::Op::Get("_no_gradient");
    for (uint32_t i = 0; i < param.num_no_grad_inputs; ++i) {
      ret.at(ret.size() - i - 1) = nnvm::NodeEntry{np, 0, 0};
    }
    for (index_t i = 0; i < param.num_states; ++i) {
      ret.emplace_back(nnvm::NodeEntry{np, 0, 0});
    }
  }
  return ret;
}


NNVM_REGISTER_OP(_backward)
.describe("backward operator of NN module")
.set_num_outputs([] (const NodeAttrs& attrs) {
  const NNBackwardParam& param = dmlc::get<NNBackwardParam>(attrs.parsed);
  return param.forward_readonly_inputs;
  })
.set_num_inputs([] (const NodeAttrs& attrs) {
  const NNBackwardParam& param = dmlc::get<NNBackwardParam>(attrs.parsed);
  uint32_t n = param.num_states + 1;
  if (param.need_inputs) n += param.forward_readonly_inputs;
  if (param.need_outputs) n += 1;
  return n;
  })
.set_attr<nnvm::FBackwardOutToInIndex>("FBackwardOutToInIndex", [](const NodeAttrs& attrs) {
  const NNBackwardParam& param = dmlc::get<NNBackwardParam>(attrs.parsed);
  std::vector<uint32_t> vec;
  for (uint32_t i = 0; i < param.forward_readonly_inputs; ++i) {
    vec.push_back(i);
  }
  return vec;
  })
.set_attr<nnvm::FBackwardInGradIndex>("FBackwardInGradIndex", [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>{0};
  });


// common attributes for nn module.
NNVM_REGISTER_OP_GROUP(nn_module)
.set_attr<FGradient>("FGradient", MakeNNBackwardNode)
.set_attr<bool>("TBackwardNeedInputs", true)
.set_attr<bool>("TBackwardNeedOutputs", true);


NNVM_REGISTER_OP_GROUP(nn_criterion)
.set_attr<FGradient>("FGradient", MakeNNBackwardNode)
.set_attr<int>("TBackwardNumNoGradInputs", 1)
.set_attr<bool>("TBackwardNeedInputs", true)
.set_attr<bool>("TBackwardNeedOutputs", false)
.set_attr<FInferShape>("FInferShape", ScalarShape);


NNVM_REGISTER_OP(softmax)
.describe("Softmax operation")
.set_num_inputs(1)
.include("nn_module")
.set_attr<FLuaCreateNNModule>(
    "FLuaCreateNNModule", R"(
function(ishape, kwarg)
  return nn.SoftMax()
end
)")
.set_attr<FInferShape>("FInferShape", SameShape);


NNVM_REGISTER_OP(relu)
.describe("Relu operation")
.set_num_inputs(1)
.include("nn_module")
.set_attr<FLuaCreateNNModule>(
    "FLuaCreateNNModule", R"(
function(ishape, kwarg)
  return nn.ReLU()
end
)")
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<bool>("TBackwardNeedOutputs", true);


NNVM_REGISTER_OP(tanh)
.describe("Tanh operation")
.set_num_inputs(1)
.include("nn_module")
.set_attr<FLuaCreateNNModule>(
    "FLuaCreateNNModule", R"(
function(ishape, kwarg)
  return nn.Tanh()
end
)")
.set_attr<FInferShape>("FInferShape", SameShape);


// same as matrix multiplication, but automatically infers shape
struct LinearParam : public dmlc::Parameter<LinearParam> {
  uint32_t num_hidden;
  bool no_bias;

  DMLC_DECLARE_PARAMETER(LinearParam) {
    DMLC_DECLARE_FIELD(num_hidden).set_default(0);
    DMLC_DECLARE_FIELD(no_bias).set_default(true);
  }
};
DMLC_REGISTER_PARAMETER(LinearParam);

inline bool LinearShape(const NodeAttrs& attrs,
                        std::vector<TShape> *ishape,
                        std::vector<TShape> *oshape) {
  const auto& param = dmlc::get<LinearParam>(attrs.parsed);
  if (ishape->at(0).ndim() == 0) return false;
  const TShape& in = ishape->at(0);
  TShape wshape;
  if (param.num_hidden != 0) {
    wshape = TShape{param.num_hidden, in[1]};
    SHAPE_ASSIGN(ishape->at(1), wshape);
  } else {
    if (ishape->at(1).ndim() == 0) return false;
  }
  if (ishape->size() > 2) {
    TShape bshape{ishape->at(1)[0]};
    SHAPE_ASSIGN(ishape->at(2), bshape);
  }
  TShape out{in[0], wshape[0]};
  SHAPE_ASSIGN(oshape->at(0), out);
  return true;
}

NNVM_REGISTER_OP(linear)
.describe("A linear transformation layer")
.set_attr_parser(ParamParser<LinearParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
    return (dmlc::get<LinearParam>(attrs.parsed).no_bias? 2 : 3);
  })
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
    if (dmlc::get<LinearParam>(attrs.parsed).no_bias) {
      return std::vector<std::string>{"data", "weight"};
    } else {
      return std::vector<std::string>{"data", "weight", "bias"};
    }
  })
.include("nn_module")
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
)")
.set_attr<FInferShape>("FInferShape", LinearShape);


struct ConvPoolParam : public dmlc::Parameter<ConvPoolParam> {
  TShape ksize;
  TShape strides;
  std::string padding;
  std::string data_format;
  bool no_bias;
  uint32_t num_filter;

  DMLC_DECLARE_PARAMETER(ConvPoolParam) {
    DMLC_DECLARE_FIELD(ksize).set_default(TShape{1, 1, 1, 1});
    DMLC_DECLARE_FIELD(strides).set_default(TShape{1, 1, 1, 1});
    DMLC_DECLARE_FIELD(padding).set_default("SAME");
    DMLC_DECLARE_FIELD(data_format).set_default("NCHW");
    DMLC_DECLARE_FIELD(no_bias).set_default(true);
    DMLC_DECLARE_FIELD(num_filter).set_default(0);
  }
};
DMLC_REGISTER_PARAMETER(ConvPoolParam);

inline bool ConvPoolShape(const NodeAttrs& attrs,
                          std::vector<TShape> *ishape,
                          std::vector<TShape> *oshape) {
  const auto& param = dmlc::get<ConvPoolParam>(attrs.parsed);
  if (ishape->at(0).ndim() == 0) return false;
  const TShape& in = ishape->at(0);
  TShape filter;
  if (ishape->size() == 1) {
    // pooling
    CHECK_EQ(param.ksize.ndim(), 4);
    CHECK(param.ksize[0] == param.ksize[3] && param.ksize[0] == 1);
    filter = TShape{in[1], in[1], param.ksize[1], param.ksize[2]};
  } else if (param.ksize.ndim() == 4 && param.num_filter != 0) {
    CHECK(param.ksize[0] == param.ksize[3] && param.ksize[0] == 1);
    filter = TShape{param.num_filter, in[1], param.ksize[1], param.ksize[2]};
    SHAPE_ASSIGN(ishape->at(1), filter);
  } else {
    if (ishape->at(1).ndim() == 0) return false;
    filter = ishape->at(1);
  }
  if (ishape->size() > 2) {
    SHAPE_ASSIGN(ishape->at(2), TShape{filter[0]});
  }
  CHECK(param.strides[0] == param.strides[3] && param.strides[0] == 1);
  uint32_t dH = param.strides[1];
  uint32_t dW = param.strides[2];
  uint32_t padW = 0, padH = 0;
  if (param.padding == "SAME") {
    padH = (filter[2] - 1) / 2;
    padW = (filter[3] - 1) / 2;
  }
  CHECK_EQ(in[1], filter[1])
      << "in=" << in << ", filter=" << filter;
  // batch, out, height, width
  oshape->at(0) = TShape{in[0], filter[0],
                         (in[2] + 2 * padH - filter[2]) / dH + 1,
                         (in[3] + 2 * padW - filter[3]) / dW + 1};
  return true;
}

NNVM_REGISTER_OP(conv2d)
.describe("Convolution operation")
.set_num_inputs([](const NodeAttrs& attrs){
    return (dmlc::get<ConvPoolParam>(attrs.parsed).no_bias? 2 : 3);
  })
.set_attr_parser(ParamParser<ConvPoolParam>)
.include("nn_module")
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
)")
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
    if (dmlc::get<ConvPoolParam>(attrs.parsed).no_bias) {
      return std::vector<std::string>{"data", "weight"};
    } else {
      return std::vector<std::string>{"data", "weight", "bias"};
    }
  })
.set_attr<FInferShape>("FInferShape", ConvPoolShape)
.set_attr<bool>("TBackwardNeedOutputs", false);


NNVM_REGISTER_OP(max_pool)
.describe("Max pooling")
.set_num_inputs(1)
.set_attr_parser(ParamParser<ConvPoolParam>)
.include("nn_module")
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
)")
.set_attr<FInferShape>("FInferShape", ConvPoolShape)
.set_attr<bool>("TBackwardNeedOutputs", false);


NNVM_REGISTER_OP(mean_sparse_softmax_cross_entropy_with_logits)
.describe("Softmax cross entropy given logit and label")
.set_num_inputs(2)
.include("nn_criterion")
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
.describe("Flatten to 2D")
.set_num_inputs(1)
.set_attr<FLuaCompute>("FLuaCompute", LuaReshape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FInferShape>(
    "FInferShape", [](const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
      const TShape& in = ishape->at(0);
      if (in.ndim() == 0) return false;
      TShape out{in[0], in.ProdShape(1, in.ndim())};
      SHAPE_ASSIGN(oshape->at(0), out);
      return true;
    })
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return MakeBackwardGrads("_flatten_backward", n,
                               {ograds[0], n->inputs[0]});
    });

NNVM_REGISTER_OP(_flatten_backward)
.set_num_inputs(1)
.set_attr<FLuaCompute>("FLuaCompute", LuaReshape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FBackwardOutToInIndex>(
    "FBackwardOutToInIndex", [](const NodeAttrs& attrs) {
      return std::vector<uint32_t>{0};
    })
.set_attr<FBackwardInGradIndex>(
    "FBackwardInGradIndex", [](const NodeAttrs& attrs) {
      return std::vector<uint32_t>{0};
    });

}  // namespace tinyflow
