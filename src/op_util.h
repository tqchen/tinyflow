/*!
 *  Copyright (c) 2016 by Contributors
 * \file op_util.h
 * \brief common util to define operators
 */
#ifndef TINYFLOW_OP_UTIL_H_
#define TINYFLOW_OP_UTIL_H_

#include <tinyflow/base.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <vector>
#include <string>
#include <utility>

namespace tinyflow {

using namespace nnvm;

// assign rhs to lhs, check if shape is consistent
#define SHAPE_ASSIGN(lhs, rhs)                                \
  if ((lhs).ndim() == 0) (lhs) = (rhs);                       \
  else                                                        \
    CHECK_EQ(lhs, rhs) << "shape inference inconsistent";     \

// assign rhs to lhs, check if type is consistent
#define DTYPE_ASSIGN(lhs, rhs)                                \
  if ((lhs) == -1) (lhs) = (rhs);                             \
  else                                                        \
    CHECK_EQ(lhs, rhs) << "type inference inconsistent";     \


// simply return the shape as same
inline bool SameShape(const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
  TShape def_v;
  for (TShape& pshape : *oshape) {
    if (pshape.ndim() != 0) {
      def_v = pshape; break;
    }
  }
  if (def_v.ndim() == 0) {
    for (TShape& pshape : *ishape) {
      if (pshape.ndim() != 0) {
        def_v = pshape;
        if (pshape.ndim() != 1 || pshape[0] != 1) {
          break;
        }
      }
    }
  }
  if (def_v.ndim() == 0) return false;

  for (TShape& pshape : *oshape) {
    SHAPE_ASSIGN(pshape, def_v);
  }
  for (TShape& pshape : *ishape) {
    if (pshape.ndim() == 1 && pshape[0] == 1) {
      continue;
    }
    SHAPE_ASSIGN(pshape, def_v);
  }
  return true;
}

// The output is a scalar.
inline bool ScalarShape(const NodeAttrs& attrs,
                        std::vector<TShape> *ishape,
                        std::vector<TShape> *oshape) {
  for (TShape& pshape : *ishape) {
    if (pshape.ndim() == 0) return false;
  }
  SHAPE_ASSIGN(oshape->at(0), TShape{1});
  return true;
}

inline std::vector<std::pair<int, int> > InplaceIn0Out0(const NodeAttrs& attrs) {
  return {{0, 0}};
}

inline std::vector<std::pair<int, int> > InplaceIn1Out0(const NodeAttrs& attrs) {
  return {{1, 0}};
}

/*! \brief Parse keyword arguments as PType arguments and save to parsed */
template<typename PType>
inline void ParamParser(nnvm::NodeAttrs* attrs) {
  PType param;
  param.Init(attrs->dict);
  attrs->parsed = std::move(param);
}

// quick helper to make node
inline NodeEntry MakeNode(const char* op_name,
                          std::string node_name,
                          std::vector<NodeEntry> inputs,
                          std::unordered_map<std::string, std::string> kwarg = {}) {
  NodePtr p = Node::Create();
  p->attrs.op = nnvm::Op::Get(op_name);
  p->attrs.name = std::move(node_name);
  p->inputs = std::move(inputs);
  p->attrs.dict = std::move(kwarg);
  if (p->op()->attr_parser != nullptr) {
    p->op()->attr_parser(&(p->attrs));
  }
  return NodeEntry{p, 0, 0};
}

// make backward convention node of an op
inline std::vector<NodeEntry> MakeBackwardGrads(
    const char* op_name,
    const NodePtr& n,
    std::vector<NodeEntry> inputs,
    std::unordered_map<std::string, std::string> kwarg = {}) {
  NodePtr p = Node::Create();
  p->attrs.op = nnvm::Op::Get(op_name);
  p->attrs.name = std::move(n->attrs.name + "_backward");
  p->inputs = std::move(inputs);
  p->attrs.dict = std::move(kwarg);
  p->control_deps.push_back(n);
  std::vector<NodeEntry> ret;
  for (uint32_t i = 0; i < p->num_outputs(); ++i) {
    ret.emplace_back(NodeEntry{p, i, 0});
  }
  return ret;
}

// special parameter stored in backward node.
struct NNBackwardParam {
  // total number of inputs in forward op
  uint32_t forward_readonly_inputs;
  // number of internal states in the op
  uint32_t num_states{0};
  // number of inputs who do not have gradients.
  uint32_t num_no_grad_inputs{0};
  // whether backward need all te inputs.
  bool need_inputs{true};
  // whether backward need all the outputs.
  bool need_outputs{true};
};

}  // namespace tinyflow

#endif  // TINYFLOW_OP_UTIL_H_
