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
#include <utility>

namespace tinyflow {

using namespace nnvm;

#define SHAPE_ASSIGN(lhs, rhs)                                \
  if ((lhs).ndim() == 0) (lhs) = (rhs);                       \
  else                                                        \
    CHECK_EQ(lhs, rhs) << "shape inference inconsistent";     \

#define DTYPE_ASSIGN(lhs, rhs)                                \
  if ((lhs) == -1) (lhs) = (rhs);                             \
  else                                                        \
    CHECK_EQ(lhs, rhs) << "shape inference inconsistent";     \


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
        def_v = pshape; break;
      }
    }
  }
  if (def_v.ndim() == 0) return false;

  for (TShape& pshape : *oshape) {
    SHAPE_ASSIGN(pshape, def_v);
  }
  for (TShape& pshape : *ishape) {
    SHAPE_ASSIGN(pshape, def_v);
  }
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

// special parameter stored in backward node.
struct BackwardParam {
  // total number of inputs in forward op
  uint32_t forward_readonly_inputs;
  // number of internal states in the op
  uint32_t num_states{0};
  // whether backward need all te inputs.
  bool need_inputs;
  // whether backward need all the outputs.
  bool need_outputs;
};

}  // namespace tinyflow

#endif  // TINYFLOW_OP_UTIL_H_
