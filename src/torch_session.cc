// Copyright (c) 2016 by Contributors
#include <tinyflow/base.h>
#include <nnvm/pass_functions.h>

#include <memory>
#include "./torch_util.h"
#include "./op_util.h"

namespace tinyflow {

using dmlc::any;
using nnvm::Graph;
using nnvm::IndexedGraph;
using nnvm::ShapeVector;
using nnvm::DTypeVector;
using nnvm::StorageVector;

class TorchExecutor;

/*! \brief shared variable */
struct VarState {
  /*! \brief The internal internal tensor */
  LuaRef tensor;
  /*! \brief The corresponding tblob */
  TBlob blob;

  /*! \return Whether the tensor is initialized already */
  inline bool initialized() const {
    return !tensor.is_nil();
  }
  // reset the space.
  inline void ResetSpace(TShape shape, int dev_mask = kCPU, int dtype = 0) {
    if (tensor.is_nil() ||
        shape != blob.shape ||
        dev_mask != blob.dev_mask ||
        dtype != blob.dtype) {
      TorchState* th = TorchState::ThreadLocalState();
      if (tensor.is_nil()) {
        tensor = th->NewTensorEmpty(dev_mask, dtype);
      }
      th->ResetStorage(
          tensor, th->NewStorage(shape.Size(), dev_mask, dtype), shape);
      this->blob = th->GetTBlob(tensor);
    }
  }
};

// shared variable map structure
using VarStateMap = std::unordered_map<std::string, std::shared_ptr<VarState> >;

// torch session.
class TorchSession : public Session {
 public:
  // simple session that binds to one device.
  explicit TorchSession(const std::string& default_device) {
    if (default_device.find("gpu") != std::string::npos) {
      default_dev_mask_ = kGPU;
    }
  }
  const std::vector<TBlob>&
  Run(nnvm::Symbol* sym,
      const std::unordered_map<std::string, TBlob>& inputs) override;

 private:
  // entry to store cached executor
  struct ExecEntry {
    std::shared_ptr<TorchExecutor> exec;
    size_t use_count{0};
  };
  int default_dev_mask_{kCPU};
  // local cached variable states.
  VarStateMap states_;
  // cached executor
  std::unordered_map<Symbol*, ExecEntry> cached_execs_;
};


class TorchExecutor {
 public:
  // initialize the executor
  // possibly update the states.
  void Init(nnvm::Symbol symbol, VarStateMap* states, int default_dev_mask);
  /// run the executor, return the outputs.
  const std::vector<TBlob>& Run(const std::unordered_map<std::string, TBlob>& inputs);
  // return corresponding internal symbol
  inline const nnvm::Symbol& symbol() const {
    return symbol_;
  }

 private:
  // setup the executor space.
  void Setup(const std::unordered_map<std::string, TBlob>& inputs);
  void SetupShapeDType(const std::unordered_map<std::string, TBlob>& inputs, bool* need_redo_infer);
  void SetupStorage();
  void SetupOpExecs();
  // internal symbol and graph
  nnvm::Symbol symbol_;
  nnvm::Graph graph_;
  // shape vector in graph attribute
  const ShapeVector* node_shape_{nullptr};
  // type vector in graph attribute
  const DTypeVector* node_dtype_{nullptr};
  // ----------------------------
  // node auxiliary data structures
  // The device of this executor
  int dev_mask_{kGPU};
  // node id of place holder ops
  std::vector<uint32_t> placeholder_nids_;
  // size of number of node, placeholder_tblobs_[nid].data != nullptr
  // if nid is a placeholder and the content is the corresponding TBlob to be copied in.
  std::vector<TBlob> placeholder_tblobs_;
  // node id of variable that is assigned in this executor
  std::vector<uint32_t> assign_var_nids_;
  // node id of variable that is readed by this executor
  // can overlap with assign_var_nids_
  std::vector<uint32_t> read_var_nids_;
  // vector maps nid->state, nullptr for non variables.
  std::vector<VarState*> node_states_;
  // ----------------------------
  // execution information
  // data of each outputs
  std::vector<LuaRef> data_entry_;
  // whether data entry is variable.
  std::vector<bool> data_entry_is_var_;
  // internal storage space.
  std::vector<LuaRef> storage_pool_;
  // operator executor closures
  std::vector<LuaRef> op_execs_;
  // lua module states of each operator.
  std::vector<LuaRef> op_exec_modules_;
  // The storage space to hold outputs.
  std::vector<LuaRef> outputs_;
  std::vector<TBlob> output_blobs_;
};

Session* Session::Create(const std::string& option) {
  return new TorchSession(option);
}

const std::vector<TBlob>& TorchSession::Run(
    nnvm::Symbol* sym,
    const std::unordered_map<std::string, TBlob>& inputs) {
  if (cached_execs_.count(sym) != 0) {
    auto& entry = cached_execs_.at(sym);
    const nnvm::Symbol& s = entry.exec->symbol();
    bool stale_exec = (s.outputs.size() != sym->outputs.size());
    if (!stale_exec) {
      for (size_t i = 0; i < s.outputs.size(); ++i) {
        if (s.outputs[i].node.get() != sym->outputs[i].node.get() ||
            s.outputs[i].index != sym->outputs[i].index ||
            s.outputs[i].version != sym->outputs[i].version) {
          stale_exec = true; break;
        }
      }
    }
    if (!stale_exec) {
      ++entry.use_count;
      return entry.exec->Run(inputs);
    } else {
      cached_execs_.erase(sym);
    }
  }
  // dump technique, remove all previous executors
  // better strategy, LRU?
  cached_execs_.clear();
  ExecEntry e;
  e.exec = std::make_shared<TorchExecutor>();
  e.exec->Init(*sym, &states_, default_dev_mask_);
  cached_execs_[sym] = e;
  return e.exec->Run(inputs);
}

void TorchExecutor::Init(nnvm::Symbol symbol,
                         VarStateMap* states,
                         int default_dev_mask) {
  dev_mask_ = default_dev_mask;
  if (dev_mask_ == kGPU) TorchState::ThreadLocalState()->InitGPU();
  graph_.outputs = symbol.outputs;
  symbol_ = std::move(symbol);
  // initialize all node auxiliary data structures.
  const Op* assign_op = Op::Get("assign");
  const Op* placeholder_op = Op::Get("placeholder");
  const auto& idx = graph_.indexed_graph();
  node_states_.resize(idx.num_nodes(), nullptr);

  std::vector<int> read_count(idx.num_nodes(), 0);
  std::vector<int> assign_count(idx.num_nodes(), 0);
  placeholder_tblobs_.resize(idx.num_nodes());

  for (uint32_t i = idx.num_nodes(); i != 0; --i) {
    uint32_t nid = i - 1;
    auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      const std::string& key = inode.source->attrs.name;
      if (states->count(key) == 0) {
        (*states)[key] = std::make_shared<VarState>();
      }
      node_states_[nid] = states->at(key).get();
      if (read_count[nid] != 0 || assign_count[nid] == 0) {
        read_var_nids_.push_back(nid);
      }
      if (assign_count[nid] != 0) {
        assign_var_nids_.push_back(nid);
      }
    } else {
      if (inode.source->op() == placeholder_op) {
        placeholder_nids_.push_back(nid);
      } else if (inode.source->op() == assign_op) {
        CHECK_EQ(inode.inputs.size(), 2);
        ++read_count[inode.inputs[1].node_id];
        ++assign_count[inode.inputs[0].node_id];
      } else {
        for (auto e : inode.inputs) {
          ++read_count[e.node_id];
        }
      }
    }
  }
}

const std::vector<TBlob>&
TorchExecutor::Run(const std::unordered_map<std::string, TBlob>& inputs) {
  Setup(inputs);
  {
    // execution
    const auto& idx = graph_.indexed_graph();
    auto* th = TorchState::ThreadLocalState();
    for (size_t i = 0; i < op_execs_.size(); ++i) {
      // copy in place holder as demanded.
      if (placeholder_tblobs_[i].data != nullptr) {
        th->CopyFromTo(th->NewTensorShared(placeholder_tblobs_[i]),
                       data_entry_[idx.entry_id(i, 0)]);
      }
      if (!op_execs_[i].is_nil()) op_execs_[i]();
    }
  }
  {
    // copy outputs
    output_blobs_.clear();
    auto* th = TorchState::ThreadLocalState();
    const auto& idx = graph_.indexed_graph();
    for (size_t i = 0; i < outputs_.size(); ++i) {
      uint32_t eid = idx.entry_id(idx.outputs()[i]);
      th->CopyFromTo(data_entry_[eid], outputs_[i]);
      output_blobs_.push_back(th->GetTBlob(outputs_[i]));
    }
  }
  return output_blobs_;
}

void TorchExecutor::Setup(const std::unordered_map<std::string, TBlob>& inputs) {
  bool need_redo_infer;
  SetupShapeDType(inputs, &need_redo_infer);
  if (need_redo_infer) SetupStorage();
  if (need_redo_infer) {
    op_execs_.clear();
    op_exec_modules_.clear();
    SetupOpExecs();
  }
  {
    // copy inputs
    const auto& idx = graph_.indexed_graph();
    for (uint32_t nid : placeholder_nids_) {
      const std::string& key = idx[nid].source->attrs.name;
      const TBlob& value = inputs.at(key);
      placeholder_tblobs_[nid] = value;
    }
  }
}

void TorchExecutor::SetupShapeDType(
    const std::unordered_map<std::string, TBlob>& inputs,
    bool* p_need_redo_infer) {
  const auto& idx = graph_.indexed_graph();
  bool& need_redo_infer = *p_need_redo_infer;
  need_redo_infer = (node_shape_ == nullptr);

  // check the variable states
  if (!need_redo_infer) {
    CHECK(node_dtype_ != nullptr);
    for (uint32_t nid : read_var_nids_) {
      VarState* state = node_states_[nid];
      CHECK(state != nullptr);
      CHECK(state->initialized())
          << "Attempt to execute a graph un-initialized Variable";
      if (node_shape_->at(nid) != state->blob.shape) {
        need_redo_infer = true; break;
      }
      if (node_dtype_->at(nid) != state->blob.dtype) {
        need_redo_infer = true; break;
      }
    }
  }
  // check placeholder shapes.
  if (!need_redo_infer) {
    for (uint32_t nid : placeholder_nids_) {
      const std::string& key = idx[nid].source->attrs.name;
      CHECK(inputs.count(key))
          << "Not enought placeholder argument to feed_dict";
      const TBlob& value = inputs.at(key);
      if (node_shape_->at(idx.entry_id(nid, 0)) != value.shape) {
        need_redo_infer = true; break;
      }
      if (node_dtype_->at(idx.entry_id(nid, 0)) != value.dtype) {
        need_redo_infer = true; break;
      }
    }
  }

  if (!need_redo_infer) return;
  // run shape inference.
  ShapeVector new_shape(idx.num_node_entries(), TShape());
  DTypeVector new_dtype(idx.num_node_entries(), -1);

  for (uint32_t nid : read_var_nids_) {
    VarState* state = node_states_[nid];
    CHECK(state->initialized())
        << "Attempt to execute a graph un-initialized Variable";
    new_shape[idx.entry_id(nid, 0)] = state->blob.shape;
    new_dtype[idx.entry_id(nid, 0)] = state->blob.dtype;
  }
  for (uint32_t nid : placeholder_nids_) {
    const std::string& key = idx[nid].source->attrs.name;
    const TBlob& value = inputs.at(key);
    new_shape[idx.entry_id(nid, 0)] = value.shape;
    new_dtype[idx.entry_id(nid, 0)] = value.dtype;
  }
  graph_.attrs["shape"] = std::make_shared<any>(std::move(new_shape));
  graph_.attrs["dtype"] = std::make_shared<any>(std::move(new_dtype));
  graph_ = ApplyPasses(std::move(graph_), {"InferShape", "InferType"});
  CHECK_EQ(graph_.GetAttr<size_t>("shape_num_unknown_nodes"), 0)
      << "Shape information in the graph is in-complete";
  CHECK_EQ(graph_.GetAttr<size_t>("dtype_num_unknown_nodes"), 0)
      << "Type information in the graph is in-complete";
  node_shape_ = &(graph_.GetAttr<ShapeVector>("shape"));
  node_dtype_ = &(graph_.GetAttr<DTypeVector>("dtype"));
  // setup out Variable space.
  for (uint32_t nid : assign_var_nids_) {
    node_states_[nid]->ResetSpace(
        node_shape_->at(idx.entry_id(nid, 0)),
        dev_mask_,
        node_dtype_->at(idx.entry_id(nid, 0)));
  }
}

void TorchExecutor::SetupStorage() {
  const auto& idx = graph_.indexed_graph();
  if (storage_pool_.size() == 0) {
    graph_ = nnvm::ApplyPass(std::move(graph_), "PlanMemory");
  }
  const auto& vstorage = graph_.GetAttr<StorageVector>("storage_id");
  const auto& vshape = graph_.GetAttr<ShapeVector>("shape");
  auto* th = TorchState::ThreadLocalState();

  if (data_entry_.size() == 0) {
    data_entry_.resize(idx.num_node_entries());
    data_entry_is_var_.resize(idx.num_node_entries(), false);
    for (size_t i = 0; i < data_entry_.size(); ++i) {
      data_entry_[i] = th->NewTensorEmpty(dev_mask_);
    }
    for (uint32_t nid : idx.input_nodes()) {
      CHECK(node_states_[nid] != nullptr);
      data_entry_[idx.entry_id(nid, 0)] = node_states_[nid]->tensor;
      data_entry_is_var_[idx.entry_id(nid, 0)] = true;
    }
  }


  // size of each storage pool entry
  std::vector<size_t> pool_entry_size;
  for (size_t i = 0; i < vshape.size(); ++i) {
    if (data_entry_is_var_[i]) continue;
    int storage_id = vstorage[i];
    size_t size = vshape[i].Size();
    CHECK_GE(storage_id, 0) << "Do not support runtime shape op yet";
    size_t sid = static_cast<size_t>(storage_id);
    if (sid >= pool_entry_size.size()) {
      pool_entry_size.resize(sid + 1, 0);
    }
    pool_entry_size[sid] = std::max(pool_entry_size[sid], size);
  }
  storage_pool_.clear();
  for (size_t i = 0; i < pool_entry_size.size(); ++i) {
    storage_pool_.push_back(
        th->NewStorage(pool_entry_size[i], dev_mask_));
  }
  // assign pooled data to entry
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    if (data_entry_is_var_[i]) continue;
    int storage_id = vstorage[i];
    th->ResetStorage(data_entry_[i], storage_pool_.at(storage_id), vshape[i]);
  }

  outputs_.resize(idx.outputs().size());
  for (size_t i = 0; i < outputs_.size(); ++i) {
    uint32_t eid = idx.entry_id(idx.outputs()[i]);
    LuaRef t = th->NewTensorEmpty(kCPU);
    th->ResetStorage(t, th->NewStorage(vshape[eid].Size(), kCPU), vshape[eid]);
    outputs_[i] = t;
  }
}

void TorchExecutor::SetupOpExecs() {
  // a slightly big function to setup execution functors
  // We can separate some logics into a new pass later.
  auto* lua = LuaState::ThreadLocalState();
  const auto& idx = graph_.indexed_graph();
  const auto& lua_create_module =
      nnvm::Op::GetAttr<FLuaCreateNNModule>("FLuaCreateNNModule");
  const auto& lua_compute_code =
      nnvm::Op::GetAttr<FLuaCompute>("FLuaCompute");
  LuaRef fremove_module_storage = lua->Eval(R"(
    return
    function(m, dev_mask)
      local empty = torch.FloatTensor()
      if dev_mask == 2 then
        m = m:cuda()
        empty = empty:cuda()
      end
      if torch.isTypeOf(m, nn.Module) then
        local W, gW = m:parameters()
        if W ~= nil then
          for i, t in ipairs(W) do
            t:set(empty)
          end
          for i, t in ipairs(gW) do
            t:set(empty)
          end
        end
      end
      return m
    end
  )");
  LuaRef fcreate_nnforward_closure = lua->Eval(R"(
    return
    function(m, input, output, weight)
      if torch.isTypeOf(m, nn.Module) then
        if m:parameters() ~= nil then
          return function()
            local W, gW = m:parameters()
            for i, t in ipairs(W) do
              t:set(weight[i])
            end
            m.output:set(output)
            m:updateOutput(input)
            if not m.output:isSetTo(output) then
              output:copy(m.output)
              m.output:set(output)
            end
          end
        else
          return function()
            m.output:set(output)
            m:updateOutput(input)
            if not m.output:isSetTo(output) then
              output:copy(m.output)
              m.output:set(output)
            end
          end
        end
      else
        target = weight[1]
        assert(torch.isTypeOf(m, nn.Criterion))
        return function()
          m.output_tensor:set(output)
          m:updateOutput(input, target)
        end
      end
    end
  )");
  LuaRef fcreate_nnbackward_closure = lua->Eval(R"(
    return
    function(m, input, output, weight, gradInput, gradOutput, gradWeight)
      if torch.isTypeOf(m, nn.Module) then
        if m:parameters() ~= nil then
          return function()
            local W, gW = m:parameters()
            for i, t in ipairs(W) do
              t:set(weight[i])
            end
            for i, t in ipairs(gW) do
              t:set(gradWeight[i])
            end
            m.output:set(output)
            m.gradInput:set(gradInput)
            m:zeroGradParameters()
            m:accGradParameters(input, gradOutput, 1)
            m:updateGradInput(input, gradOutput)
            if not m.gradInput:isSetTo(gradInput) then
              gradInput:copy(m.gradInput)
              m.gradInput:set(gradInput)
            end
            for i, t in ipairs(gW) do
              if not t:isSetTo(gradWeight[i]) then
                gradWeight[i]:copy(t)
                t:set(gradWeight)
              end
            end
          end
        else
          return function()
            m.output:set(output)
            m.gradInput:set(gradInput)
            m:updateGradInput(input, gradOutput)
            if not m.gradInput:isSetTo(gradInput) then
              gradInput:copy(m.gradInput)
              m.gradInput:set(gradInput)
            end
          end
        end
      else
        assert(torch.isTypeOf(m, nn.Criterion))
        target = weight[1]
        return function()
          m.output_tensor:set(output)
          m.gradInput:set(gradInput)
          m:updateGradInput(input, target)
          if not m.gradInput:isSetTo(gradInput) then
            gradInput:copy(m.gradInput)
            m.gradInput:set(gradInput)
          end
        end
      end
    end
  )");

  op_exec_modules_.resize(idx.num_nodes());
  // setup torch.nn modules when available.
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    std::string lua_code;
    if (lua_create_module.count(inode.source->op())) {
      lua_code = "return " + lua_create_module[inode.source->op()];
      LuaRef fcreate = lua->Eval(lua_code);
      std::vector<TShape> ishape;
      for (auto& e : inode.inputs) {
        ishape.push_back(node_shape_->at(idx.entry_id(e)));
      }
      op_exec_modules_[nid] = fremove_module_storage(
          fcreate(ishape, inode.source->attrs.dict), dev_mask_);
    }
  }

  // setup executor closure
  const Op* backward_op = Op::Get("_backward");
  op_execs_.resize(idx.num_nodes());
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    std::vector<LuaRef> in_array, out_array;
    for (const auto& e : inode.inputs) {
      in_array.push_back(data_entry_[idx.entry_id(e)]);
    }
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      out_array.push_back(data_entry_[eid]);
    }

    if (lua_compute_code.count(inode.source->op())) {
      // compute function
      std::string lua_str = "return " + lua_compute_code[inode.source->op()];
      LuaRef fcompute = lua->Eval(lua_str);
      op_execs_[nid] = fcompute(
          in_array, out_array, inode.source->attrs.dict);
    } else if (!op_exec_modules_[nid].is_nil()) {
      // nn module forward
      std::vector<LuaRef> weights;
      for (size_t i = 1; i < in_array.size(); ++i) {
        weights.push_back(in_array[i]);
      }
      op_execs_[nid] = fcreate_nnforward_closure(
          op_exec_modules_[nid], in_array[0], out_array[0], weights);
      CHECK_EQ(out_array.size(), 1) << "onnly support tensor nn module";
    } else if (inode.source->op() == backward_op) {
      // nn module backward
      CHECK_GE(inode.control_deps.size(), 1);
      const NNBackwardParam& param =
          dmlc::get<NNBackwardParam>(inode.source->attrs.parsed);
      std::vector<LuaRef> weight, gradWeight;
      LuaRef gradInput, gradOutput, input, output;
      gradInput = out_array[0];
      for (size_t i = 1; i < out_array.size(); ++i) {
        gradWeight.push_back(out_array[i]);
      }
      gradOutput = in_array[0];
      // set the non-needed to be empty tensor.
      size_t in_ptr = 1;
      if (param.need_inputs) {
        input = in_array[in_ptr];
        for (size_t i = 1; i < param.forward_readonly_inputs; ++i) {
          weight.push_back(in_array[i + in_ptr]);
        }
        in_ptr += param.forward_readonly_inputs;
      } else {
        weight.resize(param.forward_readonly_inputs);
      }
      CHECK_EQ(param.num_states, 0);
      if (param.need_outputs) {
        output = in_array[in_ptr];
      }
      op_execs_[nid] = fcreate_nnbackward_closure(
          op_exec_modules_[inode.control_deps[0]],
          input, output, weight, gradInput, gradOutput, gradWeight);
    } else {
      LOG(FATAL) << "Function FLuaCompute is not registered on "
                 << inode.source->op()->name;
    }
  }
}

}  // namespace tinyflow
