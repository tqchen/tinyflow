/*!
 *  Copyright (c) 2016 by Contributors
 * \file base.h
 * \brief Basic data structures.
 */
#ifndef TINYFLOW_BASE_H_
#define TINYFLOW_BASE_H_

#include <nnvm/base.h>
#include <nnvm/node.h>
#include <nnvm/tuple.h>
#include <nnvm/graph.h>
#include <nnvm/symbolic.h>
#include <vector>
#include <string>

namespace tinyflow {

using nnvm::Op;
using nnvm::Node;
using nnvm::Symbol;
using nnvm::TShape;

/*! \brief device mask for each device */
enum DeviceMask {
  kCPU = 1,
  kGPU = 2
};

/*! \brief data type enumeration */
enum DataType {
  kFloat32 = 0
};

/*! \brief contiguous tensor block data structure */
struct TBlob {
  /*! \brief pointer to the data */
  void* data{nullptr};
  /*! \brief shape of the tensor */
  TShape shape;
  /*! \brief device mask of the corresponding device type */
  int dev_mask{kCPU};
  /*! \brief type of the tensor */
  int dtype{kFloat32};
};

/*!
 * \brief a lua function to return closure to carry out computation of an op.
 *
 *  Signature:
 *  function(inputs, outputs, kwarg)
 *  - inputs: array of input torch.FloatTensor
 *  - outputs: array of input torch.FloatTensor
 *  - kwargs: additional table arguments passed from kwarg.
 *  - return: a lua closure, with signature function() that carrys out the computation.
 *
 *  After this function, outputs content are set to be correct value.
 *  This function cannot change the storage of inputs and outputs.
 * \note Register as FLuaCompute
 *  only one of FLuaCreateNNModule/FLuaCompute is needed.
 */
using FLuaCompute = std::string;

/*!
 * \brief a lua code to create a NN module/Criterion for an op.
 *  Only allows register normal module that takes one tensor and returns one tensor.
 *
 *  Signature:
 *  function(ishape, kwarg)
 *  - ishape: array of array, shape of input arguments, including weights
 *  - kwarg: table: str->str, additional arguments to the op.
 *  - return: a nn.Module for corresponding op.
 *
 * \note Register as FLuaCreateNNModule,
 *  only one of FLuaCreateNNModule/FLuaCompute is needed.
 */
using FLuaCreateNNModule = std::string;

/*!
 * \brief If registered and TBackwardNumNoGrad=k
 *  The last k inputs do not have gradient.
 * \note Register as TBackwardNumNoGradInputs
 */
using TBackwardNumNoGradInputs = int;

/*!
 * \brief Whether backward need weight.
 * \note Register as TBackwardNeedInputs
 */
using TBackwardNeedInputs = bool;

/*!
 * \brief Whether backward op need outputs.
 * \note Register as TBackwardNeedOutputs
 */
using TBackwardNeedOutputs = bool;

/*! \brief Executor of a graph */
class Session {
 public:
  /*!
   * \brief Run the given graph
   * \param g the graph to run.
   * \param inputs The input feed_dict mapping
   * \note The session hold the ownership of the outputs.
   *  The results are only valid before calling any functions of this session again.
   * \return The output tensors.
   */
  virtual const std::vector<TBlob>& Run(
      Symbol* g,
      const std::unordered_map<std::string, TBlob>& inputs) = 0;
  /*! \brief virtual destructor */
  virtual ~Session() {}
  /*!
   * \brief create a new session of given type.
   * \param type The type of the session.
   * \return a new created session.
   */
  static Session* Create(const std::string& type);
};

}  // namespace tinyflow

#endif  // TINYFLOW_BASE_H_
