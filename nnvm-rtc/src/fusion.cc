/*!
 *  Copyright (c) 2016 by Contributors
 * \file operator_fusion.cc
 * \brief
 */
#include <nnvm-rtc/base.h>
#include <nnvm/pass.h>
#include <nnvm/tuple.h> // TShape
#include <nnvm/op_attr_types.h> // FInferShape

namespace nnvm {
namespace rtc {
namespace {

FusionNodePtr CreateFusionNode(NodePtr n) {
  FusionNodePtr ret = FusionNode::Create();
  ret->attrs.op     = n->op();
  ret->attrs.name   = n->attrs.name;
  ret->attrs.dict   = n->attrs.dict;
  ret->attrs.parsed = n->attrs.parsed;
  ret->inputs.resize(n->num_inputs(), NodeEntry{nullptr, 0, 0});
  return ret;
}

bool IsFusible(NodePtr n1, NodePtr n2) {
  static const OpMap<bool>& ewise_map = Op::GetAttr<bool>("IsElementWise");
  static const OpMap<FCodeGen>& gen_map = Op::GetAttr<FCodeGen>("FCodeGen");
  if (n1->op() != nullptr         &&
      n2->op() != nullptr         &&
      ewise_map.count(n1->op())   &&
      ewise_map.count(n2->op())   &&
      gen_map.count(n1->op())     &&
      gen_map.count(n2->op())     &&
      n2->num_outputs() == 1) {
    return true;
  }
  return false;
}

Graph Fusion(const Graph& src) {
  std::unordered_map<const Node*, NodePtr>        mirror_map;
  std::unordered_map<const Node*, FusionNodePtr>  node_fnode;
  std::unordered_map<const Node*, FusionGraph>    node_fgraph;
  // build topo order
  std::vector<NodePtr> topo_order;
  DFSVisit(src.outputs, [&](const NodePtr& node) {
      topo_order.push_back(node);
    });

  for (auto rit = topo_order.rbegin(); rit != topo_order.rend(); ++rit) {
    // LOG(INFO) << "Current Node: " << (*rit)->attrs.name;

    std::vector<NodeEntry>& children = (*rit)->inputs;
    std::unordered_set<Node*> merged_children;
    bool need_fusion = false;
    FusionNodePtr cur_fnode = nullptr;
    for (auto it = children.begin(); it != children.end(); ++it) {
      if (IsFusible(*rit, it->node)) {
        // LOG(INFO) << "  Merge Node: " << it->node->attrs.name;
        merged_children.insert(it->node.get());
        if (need_fusion == false) {
          need_fusion = true;
          if (node_fnode.count(rit->get()) == 0) {
            cur_fnode = CreateFusionNode(*rit);
            node_fnode[rit->get()]  = cur_fnode;
          } else {
            cur_fnode = node_fnode.at(rit->get());
          }
        }
        FusionNodePtr child_fnode = CreateFusionNode(it->node);
        node_fnode[it->node.get()] = child_fnode;
        cur_fnode->inputs[it - children.begin()] = NodeEntry{child_fnode, 0, it->version+1};
      }
    }

    if (need_fusion) {
      NodePtr fusion_node;
      if (mirror_map.count(rit->get()) == 0) {
        // create a new fusion node
        fusion_node = Node::Create();
        fusion_node->attrs.op = Op::Get("fusion_op");
        static int count = 0;
        fusion_node->attrs.name = std::move("fusion" + std::to_string(count++));
        fusion_node->inputs = children;
        // LOG(INFO) << "Create Fusion Node: " << fusion_node->attrs.name;
        FusionGraph fgraph;
        fgraph.outputs.emplace_back(NodeEntry{cur_fnode, 0, 0});
        node_fgraph[fusion_node.get()] = std::move(fgraph);
      } else {
        fusion_node = mirror_map.at(rit->get());
        // LOG(INFO) << "Use Old Fusion Node: " << fusion_node->attrs.name;
      }

      // merge inputs
      std::vector<NodeEntry>& finputs = fusion_node->inputs;
      for (auto it = finputs.begin(); it != finputs.end(); ++it) {
        if (merged_children.count(it->node.get()) != 0) {
          NodePtr key = it->node;
          it = finputs.erase(it);
          int node_index = it - finputs.begin();
          finputs.insert(it, key->inputs.begin(), key->inputs.end());
          it = finputs.begin() + node_index;
        }
      }

      mirror_map[rit->get()] = fusion_node;
      for (auto it = merged_children.begin(); it != merged_children.end(); ++it) {
        if (!(*it)->is_variable()) {
          mirror_map[*it] = fusion_node;
        }
      }
    }
  }

  // remap old node with new node
  auto remap = [&](const NodePtr& n) {
    // for those are not in mirror_map, if need_map,
    // create a new node and add it to mirror_map
    if (mirror_map.count(n.get()) == 0) {
      bool need_map = false;
      NodePtr new_node = Node::Create();
      // rebuild inputs and control_deps of new node
      for (const NodeEntry& e : n->inputs) {
        if (mirror_map.count(e.node.get()) != 0) {
          need_map = true;
          new_node->inputs.emplace_back(
            NodeEntry{mirror_map.at(e.node.get()), e.index, e.version+1});
        }
        else {
          new_node->inputs.push_back(e);
        }
      }
      for (const NodePtr& e : n->control_deps) {
        if (mirror_map.count(e.get()) != 0) {
          need_map = true;
          new_node->control_deps.push_back(mirror_map.at(e.get()));
        }
        else {
          new_node->control_deps.push_back(e);
        }
      }
      if (need_map) {
        new_node->attrs.op   = n->op();
        new_node->attrs.name = n->attrs.name;
        mirror_map[n.get()] = std::move(new_node);
      }
    }
  };
  DFSVisit(src.outputs, remap);

  // update inputs and control deps of nodes which
  // are in mirror_map already, like fusion nodes
  for (auto kv : mirror_map) {
    for (auto it = kv.second->inputs.begin(); it != kv.second->inputs.end(); ++it) {
      if (mirror_map.count(it->node.get())) {
        *it = NodeEntry{mirror_map.at(it->node.get()), it->index, it->version+1};
      }
    }
    for (auto it = kv.second->control_deps.begin();
         it != kv.second->control_deps.end(); ++it) {
      if (mirror_map.count(it->get())) {
        *it = mirror_map.at(it->get());
      }
    }
  }

  // rebuild return graph
  Graph ret;
  for (const NodeEntry& e: src.outputs) {
    auto it = mirror_map.find(e.node.get());
    if (it != mirror_map.end()) {
      ret.outputs.emplace_back(NodeEntry{it->second, e.index, e.version+1});
    } else {
      ret.outputs.push_back(e);
    }
  }
  ret.attrs["fusion_graph"] = std::make_shared<any>(std::move(node_fgraph));
  return ret;
}


#define SHAPE_ASSIGN(lhs, rhs)                              \
  if ((lhs).ndim() == 0) (lhs) = (rhs);                     \
  else                                                      \
    CHECK_EQ(lhs, rhs) << "shape inference inconsistant";   \

// simply return the shape as same
inline bool FusionShape(const NodeAttrs& attrs,
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

// register pass
NNVM_REGISTER_PASS(Fusion)
.describe("fusion pass")
.set_body(Fusion)
.set_change_graph(true);

NNVM_REGISTER_OP(fusion_op)
.describe("fusion op")
.set_attr<bool>("IsElementWise", true)
.set_attr<FInferShape>("FInferShape", FusionShape);

}  // namespace
}  // namespace pass
}  // namespace nnvm
