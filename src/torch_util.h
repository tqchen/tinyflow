/*!
 *  Copyright (c) 2016 by Contributors
 * \file torch_util.h
 * \brief common util to reuse things from torch.
 */
#ifndef TINYFLOW_TORCH_UTIL_H_
#define TINYFLOW_TORCH_UTIL_H_

#include <tinyflow/base.h>
#include <dmlc/lua.h>
#include <dmlc/thread_local.h>
#include <vector>

namespace dmlc {
namespace lua_stack {
// enable pass in TShape as arguments
template<>
struct Handler<nnvm::TShape> {
  static inline nnvm::TShape Get(lua_State* L, int index, LuaState* s) {
    std::vector<uint32_t> v = Handler<std::vector<uint32_t> >::Get(L, index, s);
    return nnvm::TShape(v.begin(), v.end());
  }
  static inline void Push(lua_State* L, const nnvm::TShape& shape) {
    std::vector<uint32_t> v(shape.begin(), shape.end());
    Handler<std::vector<uint32_t> >::Push(L, v);
  }
};

}  // namespace lua_stack
}  // namespace dmlc

namespace tinyflow {

using dmlc::LuaRef;
using dmlc::LuaState;

// hhelper to create new functions
class TorchState {
 public:
  TorchState() {
    auto* lua = LuaState::ThreadLocalState();
    lua->Eval("require 'torch'");
    lua->Eval("require 'nn'");
    lua->Eval("torch.setdefaulttensortype('torch.FloatTensor')");
    LuaRef parse_tuple = lua->Eval(R"(
      return function(s, def)
        if s == nil then
          return def
        end
        t = {}
        for k in string.gmatch(s, '%d') do
          table.insert(t, tonumber(k))
        end
        return t
      end
    )");
    LuaRef zero_index_target_criterion = lua->Eval(R"(
      return function(c)
        local updateOutput = c.updateOutput
        local updateGradInput = c.updateGradInput
        c.updateOutput = function(self, input, target)
          return updateOutput(self, input, target + 1)
        end
        c.updateGradInput = function(self, input, target)
          return updateGradInput(self, input, target + 1)
        end
        return c
      end
    )");
    lua->SetGlobalField("nn_parse_tuple", parse_tuple);
    lua->SetGlobalField("nn_zero_index_target_criterion", zero_index_target_criterion);
  }
  // prepare for GPU ops
  inline void InitGPU() {
    if (gpu_init_) return;
    LOG(INFO) << "start to initialize ...";
    auto* lua = LuaState::ThreadLocalState();
    lua->Eval("require 'cutorch'");
    lua->Eval("require 'cunn'");
    lua->Eval("require 'cudnn'");
    LOG(INFO) << "finished gpu initialization...";
    gpu_init_ = true;
  }
  // create a new storage with given size
  LuaRef NewStorage(size_t size, int dev_mask = kCPU, int dtype = 0) {
    CHECK_EQ(dtype, 0) << "only float is supported so far";
    if (fstorage_new_.is_nil()) {
      auto* lua = LuaState::ThreadLocalState();
      fstorage_new_ = lua->Eval(R"(
      return
      function(size, dev_mask)
        if dev_mask == 1 then
          return torch.FloatStorage(size)
        else
          return torch.CudaTensor(size)
        end
      end
      )");
    }
    return fstorage_new_(size, dev_mask);
  }
  // create a new empty tensor container
  LuaRef NewTensorEmpty(int dev_mask = kCPU, int dtype = 0) {
    CHECK_EQ(dtype, 0) << "only float is supported so far";
    if (ftensor_new_.is_nil()) {
      auto* lua = LuaState::ThreadLocalState();
      ftensor_new_ = lua->Eval(R"(
      return
      function(dev_mask)
        if dev_mask == 1 then
          return torch.FloatTensor()
        else
          return torch.CudaTensor()
        end
      end
      )");
    }
    return ftensor_new_(dev_mask);
  }
  // create a new tensor that shares space with src
  // The memory is managed by src.
  LuaRef NewTensorShared(TBlob src) {
    CHECK_EQ(src.dtype, 0) << "only float is supported so far";
    if (ftensor_new_shared_.is_nil()) {
      auto* lua = LuaState::ThreadLocalState();
      ftensor_new_shared_ = lua->Eval(R"(
      return
      function(ptr, shape, size, dev_mask)
        local sz = torch.LongStorage(shape)
        local storage
        if dev_mask == 1 then
          storage = torch.FloatStorage(size, ptr)
          return torch.FloatTensor(storage, 1, sz)
        else
          storage = torch.CudaStorage(size, ptr)
          return torch.CudaTensor(storage, 1, sz)
        end
      end
      )");
    }
    return ftensor_new_shared_(
        reinterpret_cast<intptr_t>(src.data),
        src.shape, src.shape.Size(), src.dev_mask);
  }
  // copy from one tensor to another one
  void CopyFromTo(LuaRef from, LuaRef to) {
    if (fcopy_from_to_.is_nil()) {
      auto* lua = LuaState::ThreadLocalState();
      fcopy_from_to_ = lua->Eval(R"(
      return
      function(from, to)
        to:copy(from)
      end
      )");
    }
    fcopy_from_to_(from, to);
  }
  // reset the storage of tensor to storage.
  void ResetStorage(LuaRef tensor,
                    LuaRef storage,
                    TShape shape) {
    if (ftensor_set_.is_nil()) {
      auto* lua = LuaState::ThreadLocalState();
      ftensor_set_ = lua->Eval(R"(
      return
      function(tensor, storage, shape)
        sz = torch.LongStorage(shape)
        -- cutorch does not support pass size in set
        tensor:set(storage, 1)
        tensor:resize(sz)
      end
      )");
    }
    ftensor_set_(tensor, storage, shape);
  }
  // Get the internal TBlob representation of
  // The tensor object must stay alive to keep the space valid.
  TBlob GetTBlob(LuaRef tensor) {
    if (fget_internal_.is_nil()) {
      auto* lua = LuaState::ThreadLocalState();
      fget_internal_ = lua->Eval(R"(
      return
      function(tensor)
        local dev_mask
        t = tensor:type()
        if t == 'torch.FloatTensor' then
          dev_mask = 1
        elseif t == 'torch.CudaTensor' then
          dev_mask = 2
        else
          error('only float tensor is supported')
        end
        local data = tonumber(torch.data(tensor, true))
        local shape =  tensor:size():totable()
        return {data, shape, dev_mask}
      end
      )");
    }
    LuaRef temp = fget_internal_(tensor);
    TBlob ret;
    ret.data = reinterpret_cast<void*>(temp[1].Get<intptr_t>());
    ret.shape = temp[2].Get<TShape>();
    ret.dev_mask = temp[3].Get<int>();
    return ret;
  }
  // return threadlocal state for torch.
  static TorchState* ThreadLocalState() {
    return dmlc::ThreadLocalStore<TorchState>::Get();
  }

 private:
  bool gpu_init_{false};
  LuaRef fstorage_new_;
  LuaRef ftensor_new_;
  LuaRef ftensor_new_shared_;
  LuaRef ftensor_set_;
  LuaRef fcopy_from_to_;
  LuaRef fget_internal_;
};

}  // namespace tinyflow

#endif  // TINYFLOW_TORCH_UTIL_H_
