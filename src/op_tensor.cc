// Copyright (c) 2016 by Contributors
// implementation of common tensor operators
#include <tinyflow/base.h>
#include <dmlc/parameter.h>
#include <nnvm/op_attr_types.h>
#include <utility>
#include "./op_util.h"

namespace tinyflow {

// shape given the ZeroParam
using namespace nnvm;

// shape parameter for zeros, ones
struct ZeroParam : public dmlc::Parameter<ZeroParam> {
  TShape shape;
  int dtype;
  DMLC_DECLARE_PARAMETER(ZeroParam) {
    DMLC_DECLARE_FIELD(shape).set_default(TShape());
    DMLC_DECLARE_FIELD(dtype).set_default(kFloat32);
  }
};
DMLC_REGISTER_PARAMETER(ZeroParam);

inline bool ZeroShape(const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
  const TShape& ts = dmlc::get<ZeroParam>(attrs.parsed).shape;
  if (ts.ndim() != 0) {
    SHAPE_ASSIGN(oshape->at(0), ts);
    return true;
  } else {
    return false;
  }
}

inline bool ZeroType(const NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
  int dtype = dmlc::get<ZeroParam>(attrs.parsed).dtype;
  DTYPE_ASSIGN(oattr->at(0), dtype);
  return true;
}


NNVM_REGISTER_OP(zeros)
.describe("zeros")
.set_num_inputs(0)
.set_attr_parser(ParamParser<ZeroParam>)
.set_attr<FInferShape>("FInferShape", ZeroShape)
.set_attr<FInferType>("FInferType", ZeroType)
.set_attr<FLuaCompute>(
    "FLuaCompute", R"(
function(x, y, kwarg)
  return function() y[1]:fill(0) end
end
)");


NNVM_REGISTER_OP(ones)
.describe("ones")
.set_num_inputs(0)
.set_attr_parser(ParamParser<ZeroParam>)
.set_attr<FInferShape>("FInferShape", ZeroShape)
.set_attr<FInferType>("FInferType", ZeroType)
.set_attr<FLuaCompute>(
    "FLuaCompute", R"(
function(x, y, kwarg)
  return function()
    y[1]:fill(1)
  end
end
)");


NNVM_REGISTER_OP(normal)
.describe("normal distribution")
.set_num_inputs(0)
.set_attr_parser(ParamParser<ZeroParam>)
.set_attr<FInferShape>("FInferShape", ZeroShape)
.set_attr<FInferType>("FInferType", ZeroType)
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
.describe("Equal comparitor")
.set_num_inputs(2)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FLuaCompute>(
        "FLuaCompute", R"(
function(x, y, kwarg)
  return function()
    y[1]:copy(torch.eq(x[1], x[2]))
  end
end
)");


NNVM_REGISTER_OP(ones_like)
.describe("ones_like")
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FLuaCompute>(
    "FLuaCompute", R"(
function(x, y, kwarg)
  return function()
    y[1]:fill(1)
  end
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
.set_attr<FLuaCompute>(
    "FLuaCompute", R"(
function(x, y, kwarg)
  return function()
    torch.add(y[1], x[1], x[2])
  end
end
)");


NNVM_REGISTER_OP(mul)
.add_alias("__mul_symbol__")
.describe("add two data together")
.set_num_inputs(2)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FLuaCompute>(
    "FLuaCompute", R"(
function(x, y, kwarg)
  return function()
    torch.cmul(y[1], x[1], x[2])
  end
end
)")
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{
        MakeNode("mul", n->attrs.name + "_grad_0",
                 {ograds[0], n->inputs[1]}),
        MakeNode("mul", n->attrs.name + "_grad_1",
                 {ograds[0], n->inputs[0]})
            };
    });


NNVM_REGISTER_OP(div)
.add_alias("__div_symbol__")
.describe("do division")
.set_num_inputs(2)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FLuaCompute>(
    "FLuaCompute", R"(
function(x, y, kwarg)
  return function()
    torch.cdiv(y[1], x[1], x[2])
  end
end
)");


NNVM_REGISTER_OP(__mul_scalar__)
.describe("Multiply symbol with scalar")
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FLuaCompute>(
    "FLuaCompute", R"(
function(x, y, kwarg)
  local scalar = tonumber(kwarg.scalar)
  return function()
    torch.mul(y[1], x[1], scalar)
  end
end
)")
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{
        MakeNode("__mul_scalar__", n->attrs.name + "_grad_0",
                 {ograds[0]}, {{"scalar", n->attrs.dict["scalar"]}}),
            };
    });


NNVM_REGISTER_OP(log)
.describe("take elemtnwise logarithm")
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FLuaCompute>(
    "FLuaCompute", R"(
function(x, y, kwarg)
  return function()
    torch.log(y[1], x[1])
  end
end
)")
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
        MakeNode("__div_symbol__", n->attrs.name + "_grad_0",
                 {ograds[0], n->inputs[0]})
      };
    });


NNVM_REGISTER_OP(matmul)
.describe("Matrix multiplication")
.set_num_inputs(2)
.set_attr<FInferShape>(
    "FInferShape", [](const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
      if (ishape->at(0).ndim() == 0) return false;
      if (ishape->at(1).ndim() == 0) return false;
      CHECK_EQ(ishape->at(0).ndim(), 2);
      CHECK_EQ(ishape->at(1).ndim(), 2);
      CHECK_EQ(ishape->at(0)[1], ishape->at(1)[0]);
      TShape target{ishape->at(0)[0], ishape->at(1)[1]};
      SHAPE_ASSIGN(oshape->at(0), target);
      return true;
    })
.set_attr<FLuaCompute>(
    "FLuaCompute", R"(
function(x, y, kwarg)
  return function()
    torch.mm(y[1], x[1], x[2])
  end
end
)")
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return MakeBackwardGrads("_matmul_backward", n,
                               {ograds[0], n->inputs[0], n->inputs[1]});
    });


// simply register a bulk op for backward
NNVM_REGISTER_OP(_matmul_backward)
.set_num_inputs(3)
.set_num_outputs(2)
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
)")
.set_attr<FBackwardOutToInIndex>(
    "FBackwardOutToInIndex", [](const NodeAttrs& attrs) {
      return std::vector<uint32_t>{0, 1};
    })
.set_attr<FBackwardInGradIndex>(
    "FBackwardInGradIndex", [](const NodeAttrs& attrs) {
      return std::vector<uint32_t>{0};
    })
.set_attr<FInplaceOption>(
    "FInplaceOption", [](const NodeAttrs& attrs) {
      // lhs->gradLhs
      return std::vector<std::pair<int, int> >{{1, 0}};
    });


struct ReduceParam : public dmlc::Parameter<ReduceParam> {
  Tuple<int> reduction_indices;
  DMLC_DECLARE_PARAMETER(ReduceParam) {
    DMLC_DECLARE_FIELD(reduction_indices).set_default(Tuple<int>());
  }
};
DMLC_REGISTER_PARAMETER(ReduceParam);


inline bool ReduceShape(const NodeAttrs& attrs,
                        std::vector<TShape> *ishape,
                        std::vector<TShape> *oshape) {
  const auto& axis
      = dmlc::get<ReduceParam>(attrs.parsed).reduction_indices;
  if (ishape->at(0).ndim() == 0) return false;
  if (axis.ndim() == 0) {
    SHAPE_ASSIGN(oshape->at(0), TShape{1});
  } else {
    TShape tmp = ishape->at(0);
    for (uint32_t idx : axis) {
      tmp[idx] = 0;
    }
    std::vector<uint32_t> ret;
    for (uint32_t x : tmp) {
      if (x != 0) ret.push_back(x);
    }
    if (ret.size() == 0) ret.push_back(1);
    SHAPE_ASSIGN(oshape->at(0), TShape(ret.begin(), ret.end()));
  }
  return true;
}


NNVM_REGISTER_OP(reduce_sum)
.describe("reduce sum")
.set_attr_parser(ParamParser<ReduceParam>)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", ReduceShape)
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
)")
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return MakeBackwardGrads("_reduce_sum_backward", n,
                               {ograds[0]}, n->attrs.dict);
    });


NNVM_REGISTER_OP(reduce_mean)
.describe("reduce mean")
.set_attr_parser(ParamParser<ReduceParam>)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", ReduceShape)
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
)")
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return MakeBackwardGrads("_reduce_mean_backward", n,
                               {ograds[0]}, n->attrs.dict);
    });


NNVM_REGISTER_OP_GROUP(ReduceBackwardIndeAttr)
.set_attr<FBackwardOutToInIndex>(
    "FBackwardOutToInIndex", [](const NodeAttrs& attrs) {
      return std::vector<uint32_t>{0};
    })
.set_attr<FBackwardInGradIndex>(
    "FBackwardInGradIndex", [](const NodeAttrs& attrs) {
      return std::vector<uint32_t>{0};
    });


NNVM_REGISTER_OP(_reduce_sum_backward)
.set_num_inputs(1)
.set_num_outputs(1)
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
)")
.include("ReduceBackwardIndeAttr");


NNVM_REGISTER_OP(_reduce_mean_backward)
.set_num_inputs(1)
.set_num_outputs(1)
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
)")
.include("ReduceBackwardIndeAttr");

NNVM_REGISTER_OP(_argmax)
.set_attr_parser(ParamParser<ReduceParam>)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", ReduceShape)
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

}  // namespace tinyflow
