/*!
 *  Copyright (c) 2015 by Contributors
 * \file rtc.cc
 * \brief wrapper for NVRTC
 */
#include <nnvm-rtc/rtc.h>
#include <iostream>

namespace nnvm {
namespace rtc {

const char RTC::str_type[] = "float";
std::unordered_map<std::string, char*> RTC::kernel_registry;


RTC::RTC(const std::string& name, const std::string& kernel) {
  name_   = name;
  kernel_ = kernel;
  if (RTC::kernel_registry.find(kernel_) != RTC::kernel_registry.end()) {
    ptx_ = RTC::kernel_registry.at(kernel_);
  } else {
    ptx_ = compile(name, kernel_);
    RTC::kernel_registry[kernel_] = ptx_;
  }
}


void RTC::Run(std::vector<void*> const& input,
              std::vector<void*> const& output,
              uint32_t num_elements) {
  static const int kBaseThreadBits = 8;
  static const int kBaseThreadNum  = 1 << kBaseThreadBits;
  static const int kMaxGridNum     = 65535;

  int num_block = (num_elements + kBaseThreadNum - 1) / kBaseThreadNum;
  if (num_block < kMaxGridNum) {
    return Run(input, output, num_elements, num_block, 1, 1, kBaseThreadNum, 1, 1);
  } else {
    LOG(FATAL) << "too large kernel, please schedule with gridDim and blockDim manually";
  }
}


void RTC::Run(std::vector<void*> const& input,
              std::vector<void*> const& output,
              uint32_t num_elements,
              uint32_t grid_dim_X,
              uint32_t grid_dim_Y,
              uint32_t grid_dim_Z,
              uint32_t block_dim_X,
              uint32_t block_dim_Y,
              uint32_t block_dim_Z) {
  CHECK(output.size());

  CUdevice cuDevice;
  CUcontext context;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));

  if (func_ == nullptr) {
    CUmodule module;
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx_, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&func_, module, name_.c_str()));
  }

  std::vector<void*> args;
  for (auto& i : input)  args.push_back(i);
  for (auto& i : output) args.push_back(i);
  args.push_back(&num_elements);

  CUDA_SAFE_CALL(cuLaunchKernel(func_,
                                grid_dim_X, grid_dim_Y, grid_dim_Z,
                                block_dim_X, block_dim_Y, block_dim_Z,
                                0, NULL, args.data(), 0));
  CUDA_SAFE_CALL(cuCtxSynchronize());
}


char* RTC::compile(const std::string& name, const std::string& code) {
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,
                                       code.c_str(),
                                       (name+".cu").c_str(),
                                       0,
                                       NULL,
                                       NULL));
    nvrtcResult compile_res = nvrtcCompileProgram(prog, 0, NULL);
    size_t log_size;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));
    char *log = new char[log_size];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
    CHECK_EQ(compile_res, NVRTC_SUCCESS) << log;

    size_t ptx_size;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptx_size));
    char *ptx = new char[ptx_size];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    return ptx;
}

}  // namespace rtc
}  // namespace nnvm
