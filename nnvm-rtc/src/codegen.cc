/*!
 *  Copyright (c) 2016 by Contributors
 * \file codegen.cc
 * \brief implementation of code generation
 */
#include <nnvm-rtc/base.h>
#include <nnvm-rtc/ast.h>
#include <nnvm/pass.h>
#include <iostream>
#include <fstream>
// TODO(ziheng) stream header file should be removed after debug

namespace nnvm {
namespace rtc {
namespace {

using ASTPtrIter = std::vector<ASTPtr>::const_iterator;


uint32_t GetVariableNum(const FusionNodePtr fnode) {
  uint32_t num = 0;
  for (auto it = fnode->inputs.begin(); it != fnode->inputs.end(); ++it) {
    if (it->node == nullptr) {
      ++num;
    } else {
      num += GetVariableNum(it->node);
    }
  }
  return num;
}


ASTPtr GenASTPtr(const FusionNodePtr fnode, ASTPtrIter begin, ASTPtrIter end) {
  static const OpMap<FCodeGen>& gen_map = Op::GetAttr<FCodeGen>("FCodeGen");
  FCodeGen gen = gen_map[fnode->op()];
  std::vector<ASTPtr> cur_inputs;
  auto it = begin;
  for (auto jt = fnode->inputs.begin(); jt != fnode->inputs.end(); ++jt) {
    if (it >= end) {
      LOG(FATAL) << "CodeGen inputs string short";
    }
    if (jt->node == nullptr) {
      cur_inputs.push_back(*it);
      ++it;
    } else {
      uint32_t num = GetVariableNum(jt->node);
      cur_inputs.push_back(GenASTPtr(jt->node, it, it + num));
      it += num;
    }
  }
  return gen(fnode, cur_inputs)[0];
}


void PrintFNode(const FusionNodePtr fnode, int indent=0) {
  std::string s = "";
  for (int i = 0; i < indent; ++i) {
    s += "  ";
  }
  std::cout << s;
  if (indent == 0) {
  } else {
    std::cout << "| ";
  }
  std::cout << fnode->attrs.name << std::endl;

  for (auto jt = fnode->inputs.begin(); jt != fnode->inputs.end(); ++jt) {
    if (jt->node == nullptr) {
      std::cout << s << "  | nullptr" << std::endl;
    } else {
      PrintFNode(jt->node, indent + 1);
    }
  }
}


Kernel KernelCodeGen(const std::string& kernel_name, const FusionGraph& fgraph) {
  const std::string type_str = "float";
  // for now, we assume fusion graph only have one output.
  FusionNodePtr fnode = fgraph.outputs[0].node;
  // PrintFNode(fnode);
  uint32_t num = GetVariableNum(fnode);

  std::string arg_str = "(";
  for (uint32_t i = 0; i < num; ++i) {
    arg_str += "const " + type_str + " *x" + std::to_string(i) + ", ";
  }
  arg_str += type_str + " *y, ";
  arg_str += "const unsigned int num_elements)";

  std::vector<ASTPtr> inputs;
  for (uint32_t i = 0; i < num; ++i) {
    ASTPtr x = ASTPtr(new VariableAST("x" + std::to_string(i)));
    ASTPtr global_idx = ASTPtr(new VariableAST("global_idx"));
    ASTPtr ast = ASTPtr(new ArraySubscriptAST(x, global_idx));
    inputs.push_back(ast);
  }
  std::string exp_str = GenASTPtr(fnode, inputs.begin(), inputs.end())->CodeGen();

  std::string kernel_str =
    "extern \"C\" __global__ void " + kernel_name + arg_str + " {\n" +
    "  unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  if (global_idx < num_elements) {\n"
    "    y[global_idx] = " + exp_str + ";\n" +
    "  }\n"
    "}";
  std::ofstream file;
  file.open(kernel_name + ".cu");
  file << kernel_str;
  file.close();
  return Kernel(kernel_name, kernel_str);
}


Graph CodeGen(Graph ret) {
  std::unordered_map<uint32_t, Kernel> kernel_map;
  const std::unordered_map<const Node*, FusionGraph>* node_fgraph =
    &(ret.GetAttr<std::unordered_map<const Node*, FusionGraph>>("fusion_graph"));

  const IndexedGraph& idx = ret.indexed_graph();
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->op() != nullptr && node_fgraph->count(node) != 0) {
      kernel_map[nid] = KernelCodeGen(node->attrs.name, node_fgraph->at(node));
    }
  }
  ret.attrs["kernel"] = std::make_shared<any>(std::move(kernel_map));
  return ret;
}


// register pass
NNVM_REGISTER_PASS(CodeGen)
.describe("TODO")
.set_body(CodeGen)
.set_change_graph(true);

}  // namespace
}  // namespace rtc
}  // namespace nnvm
