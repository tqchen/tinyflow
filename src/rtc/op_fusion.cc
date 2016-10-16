// Copyright (c) 2016 by Contributors
// implementation of operators FCodeGen attribute
#include <tinyflow/base.h>
#include <nnvm-rtc/base.h>
#include <nnvm-rtc/ast.h>
#include <vector>

using nnvm::NodePtr;
using nnvm::rtc::FCodeGen;
using nnvm::rtc::ASTPtr;
using nnvm::rtc::NumberAST;
using nnvm::rtc::CallAST;

namespace tinyflow {
NNVM_REGISTER_OP(__add_symbol__)
.set_attr<FCodeGen>(
  "FCodeGen", [](const NodePtr& n,
    const std::vector<ASTPtr>& inputs) {
    return std::vector<ASTPtr>{
      inputs[0] + inputs[1],
    };
  }
);

NNVM_REGISTER_OP(__sub_symbol__)
.set_attr<FCodeGen>(
  "FCodeGen", [](const NodePtr& n,
    const std::vector<ASTPtr>& inputs) {
    return std::vector<ASTPtr>{
      inputs[0] - inputs[1],
    };
  }
);

NNVM_REGISTER_OP(__mul_symbol__)
.set_attr<FCodeGen>(
  "FCodeGen", [](const NodePtr& n,
    const std::vector<ASTPtr>& inputs) {
    return std::vector<ASTPtr>{
      inputs[0] * inputs[1],
    };
  }
);

NNVM_REGISTER_OP(__div_symbol__)
.set_attr<FCodeGen>(
  "FCodeGen", [](const NodePtr& n,
    const std::vector<ASTPtr>& inputs) {
    return std::vector<ASTPtr>{
      inputs[0] / inputs[1],
    };
  }
);

NNVM_REGISTER_OP(__mul_scalar__)
.set_attr<FCodeGen>(
  "FCodeGen", [](const NodePtr& n,
    const std::vector<ASTPtr>& inputs) {
    double val = std::stod(n->attrs.dict["scalar"]);
    ASTPtr num_ast = ASTPtr(new NumberAST(val));
    return std::vector<ASTPtr>{
      num_ast * inputs[0],
    };
  }
);

NNVM_REGISTER_OP(exp)
.set_attr<FCodeGen>(
  "FCodeGen", [](const NodePtr& n,
    const std::vector<ASTPtr>& inputs) {
    return std::vector<ASTPtr>{
      ASTPtr(new CallAST("exp", inputs)),
    };
  }
);

}  // namespace tinyflow
