/*!
 *  Copyright (c) 2016 by Contributors
 * \file rtc.h
 * \brief wrapper for NVRTC
 */
#ifndef NNVM_RTC_RTC_H_
#define NNVM_RTC_RTC_H_
#include "./base.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <utility>

#define CUDA_SAFE_CALL(x)                                               \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS) {                                       \
      const char *msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      dmlc::LogMessageFatal(__FILE__, __LINE__).stream()                \
          << "CudaError: " #x " failed with error: " << msg;            \
    }                                                                   \
  }

#define CUDA_RUNTIME_SAFE_CALL(x)                                       \
  {                                                                     \
    cudaError_t result = x;                                             \
    if (result != cudaSuccess) {                                        \
      dmlc::LogMessageFatal(__FILE__, __LINE__).stream()                \
          << "CudaError: " #x " failed with error: "                    \
          << cudaGetErrorString(result);                                \
    }                                                                   \
  }

#define NVRTC_SAFE_CALL(x)                                              \
  {                                                                     \
    nvrtcResult result = x;                                             \
    if (result != NVRTC_SUCCESS) {                                      \
      dmlc::LogMessageFatal(__FILE__, __LINE__).stream()                \
          << "NvrtcError: " #x " failed with error: "                   \
          << nvrtcGetErrorString(result);                               \
    }                                                                   \
  }

namespace nnvm {
namespace rtc {

typedef unsigned index_t;

/*!
 * \brief Runtime compile of cuda kernel code with NVRTC
 */
class RTC {
 public:
  /*!
   * \brief Build a new kernel.
   *
   * If the same kernel has been compiled before it will be load from
   * cache instead of compile again.
   * \param name name of the kernel function.
   * \param kernel cuda code.
   */
  RTC(const std::string& name, const std::string& kernel);
  /*!
   * \brief launch a kernel for flat tensor with the engine.
   * \param input list of input CUdeviceptr address.
   * \param output list of output CUdeviceptr address.
   * \param num_elements number of elements.
   */
  void Run(std::vector<void*> const& input,
           std::vector<void*> const& output,
           uint32_t num_elements);
  /*!
   * \brief launch a kernel for flat tensor with the engine.
   * \param input list of input CUdeviceptr address.
   * \param output list of output CUdeviceptr address.
   * \param grid_dim_X kernel grid dimensions.
   * \param grid_dim_Y kernel grid dimensions.
   * \param grid_dim_Z kernel grid dimensions.
   * \param block_dim_X kernel block dimensions.
   * \param block_dim_Y kernel block dimensions.
   * \param block_dim_Z kernel block dimensions.
   */
  void Run(std::vector<void*> const& input,
           std::vector<void*> const& output,
           uint32_t num_elements,
           uint32_t grid_dim_X,
           uint32_t grid_dim_Y,
           uint32_t grid_dim_Z,
           uint32_t block_dim_X,
           uint32_t block_dim_Y,
           uint32_t block_dim_Z);

 private:
  static const char str_type[];
  static std::unordered_map<std::string, char*> kernel_registry;

  std::string name_;
  std::string kernel_;
  char* ptx_;
  CUfunction func_{nullptr};

  /*!
   * \brief compile the kernel with nvrtc.
   */
  char* compile(const std::string& name, const std::string& code);
};

}  // namespace rtc
}  // namespace nnvm

#endif  // NNVM_RTC_RTC_H_
