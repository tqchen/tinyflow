// Copyright (c) 2016 by Contributors
// implementation of common tensor operators
#include <tinyflow/base.h>
#include <dmlc/parameter.h>
#include <nnvm/op_attr_types.h>
#include <cmath>
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


NNVM_REGISTER_OP_GROUP(ElementwiseOpAttr)
.set_attr<bool>("IsElementWise", true)
.set_attr<FInferShape>("FInferShape", SameShape);


NNVM_REGISTER_OP(zeros)
.describe("zeros")
.set_num_inputs(0)
.set_attr_parser(ParamParser<ZeroParam>)
.set_attr<FInferShape>("FInferShape", ZeroShape)
.set_attr<FInferType>("FInferType", ZeroType);

NNVM_REGISTER_OP(zeros_like)
.describe("zeros_like")
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", SameShape);

NNVM_REGISTER_OP(ones)
.describe("ones")
.set_num_inputs(0)
.set_attr_parser(ParamParser<ZeroParam>)
.set_attr<FInferShape>("FInferShape", ZeroShape)
.set_attr<FInferType>("FInferType", ZeroType);


NNVM_REGISTER_OP(ones_like)
.describe("ones_like")
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", SameShape);


NNVM_REGISTER_OP(normal)
.describe("normal distribution")
.set_num_inputs(0)
.set_attr_parser(ParamParser<ZeroParam>)
.set_attr<FInferShape>("FInferShape", ZeroShape)
.set_attr<FInferType>("FInferType", ZeroType);


NNVM_REGISTER_OP(equal)
.describe("Equal comparitor")
.set_num_inputs(2)
.set_attr<FInferShape>("FInferShape", SameShape);


NNVM_REGISTER_OP(__ewise_sum__)
.describe("ewise sum")
.set_num_inputs(nnvm::kVarg)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>(n->num_inputs(), ograds[0]);
});


NNVM_REGISTER_OP(__add_symbol__)
.describe("add two data together")
.set_num_inputs(2)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{ograds[0], ograds[0]};
});


NNVM_REGISTER_OP(__add_scalar__)
.describe("add symbol with scalar")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{ograds[0]};
});


NNVM_REGISTER_OP(__sub_symbol__)
.describe("do subtract")
.set_num_inputs(2)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{
        MakeNode("__mul_scalar__", n->attrs.name + "_grad_0",
                 {ograds[0]}, {{"scalar", "1"}}),
        MakeNode("__mul_scalar__", n->attrs.name + "_grad_1",
                 {ograds[0]}, {{"scalar", "-1"}}),
      };
});


NNVM_REGISTER_OP(__sub_scalar__)
.describe("subtract symbol with scalar")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{ograds[0]};
});


NNVM_REGISTER_OP(__rsub_scalar__)
.describe("subtract scalar with symbol")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{
        MakeNode("__mul_scalar__", n->attrs.name + "_grad_1",
                 {ograds[0]}, {{"scalar", "-1"}}),
        };
});


NNVM_REGISTER_OP(mul)
.add_alias("__mul_symbol__")
.describe("add two data together")
.set_num_inputs(2)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
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


NNVM_REGISTER_OP(__mul_scalar__)
.describe("Multiply symbol with scalar")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{
        MakeNode("__mul_scalar__", n->attrs.name + "_grad_0",
                 {ograds[0]}, {{"scalar", n->attrs.dict["scalar"]}}),
      };
});


NNVM_REGISTER_OP(__div_symbol__)
.add_alias("div")
.describe("do division")
.set_num_inputs(2)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      NodeEntry n1 = MakeNode("mul", n->attrs.name + "_grad_sub_0",
                              {ograds[0], n->inputs[0]});
      NodeEntry n2 = MakeNode("__mul_scalar__", n->attrs.name + "_grad_sub_1",
                              {n1}, {{"scalar", "-1"}});
      NodeEntry n3 = MakeNode("mul", n->attrs.name + "_grad_sub_2",
                              {n->inputs[1], n->inputs[1]});
      return std::vector<NodeEntry>{
        MakeNode("__div_symbol__", n->attrs.name + "_grad_0",
                 {ograds[0], n->inputs[1]}),
        MakeNode("__div_symbol__", n->attrs.name + "_grad_1",
                 {n2, n3})
      };
});


NNVM_REGISTER_OP(__div_scalar__)
.describe("division symbol with scalar")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{
        MakeNode("__div_scalar__", n->attrs.name + "_grad_0",
                 {ograds[0]}, {{"scalar", n->attrs.dict["scalar"]}}),
      };
});


NNVM_REGISTER_OP(exp)
.describe("take elemtnwise exponation")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
        MakeNode("__mul_symbol__", n->attrs.name + "_grad_0",
                 {ograds[0], NodeEntry{n, 0, 0}})
      };
});


NNVM_REGISTER_OP(log)
.describe("take elemtnwise logarithm")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
        MakeNode("__div_symbol__", n->attrs.name + "_grad_0",
                 {ograds[0], n->inputs[0]})
      };
});


NNVM_REGISTER_OP(sqrt)
.describe("return square root of input")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    // 1 / (2 * sqrt(x)) == 1 / (2 * y)
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      NodeEntry n1 = MakeNode("__mul_scalar__", n->attrs.name + "_grad_sub_1",
                              {NodeEntry{n, 0, 0}}, {{"scalar", "2"}});
      return std::vector<NodeEntry>{
        MakeNode("__div_symbol__", n->attrs.name + "_grad_0",
                 {ograds[0], n1})
      };
});


NNVM_REGISTER_OP(__pow_symbol__)
.add_alias("pow")
.describe("take elmtnwise power between two tensor")
.set_num_inputs(2)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      // lhs: b*pow(a, b-1), rhs: pow(a, b)*ln(a)
      NodeEntry n0 = MakeNode("__add_scalar__", n->attrs.name + "_grad_sub_0",
                              {n->inputs[1]}, {{"scalar", "-1"}});
      NodeEntry n1 = MakeNode("pow", n->attrs.name + "_grad_sub_1",
                              {n->inputs[0], n0});
      NodeEntry d_lhs = MakeNode("mul", n->attrs.name + "_grad_sub_2",
                                 {n1, n->inputs[1]});
      NodeEntry n2 = MakeNode("log", n->attrs.name + "_grad_sub_3",
                              {n->inputs[0]});
      NodeEntry d_rhs = MakeNode("mul", n->attrs.name + "_grad_sub_4",
                                 {NodeEntry{n, 0, 0}, n2});
      return std::vector<NodeEntry>{
        MakeNode("__mul_symbol__", n->attrs.name + "_grad_0",
                 {ograds[0], d_lhs}),
        MakeNode("__mul_symbol__", n->attrs.name + "_grad_1",
                 {ograds[0], d_rhs})
      };

});


NNVM_REGISTER_OP(__rpow_scalar__)
.describe("take elmtnwise power between a number and a tensor")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      // pow(m, x) * ln(m)
      double num = std::stod(n->attrs.dict["scalar"]);
      NodeEntry n0 = MakeNode("__mul_scalar__", n->attrs.name + "_grad_sub_4",
                              {NodeEntry{n, 0, 0}}, {{"scalar", std::to_string(std::log(num))}});
      return std::vector<NodeEntry>{
        MakeNode("__mul_symbol__", n->attrs.name + "_grad_0",
                 {ograds[0], n0})
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
.set_attr<nnvm::TIsBackward>("TIsBackward", true);

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
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return MakeBackwardGrads("_reduce_mean_backward", n,
                               {ograds[0]}, n->attrs.dict);
    });


NNVM_REGISTER_OP_GROUP(ReduceBackwardIndeAttr)
.set_attr<nnvm::TIsBackward>("TIsBackward", true);


NNVM_REGISTER_OP(_reduce_sum_backward)
.set_num_inputs(1)
.set_num_outputs(1)
.include("ReduceBackwardIndeAttr");


NNVM_REGISTER_OP(_reduce_mean_backward)
.set_num_inputs(1)
.set_num_outputs(1)
.include("ReduceBackwardIndeAttr");


NNVM_REGISTER_OP(_argmax)
.set_attr_parser(ParamParser<ReduceParam>)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", ReduceShape);

}  // namespace tinyflow
