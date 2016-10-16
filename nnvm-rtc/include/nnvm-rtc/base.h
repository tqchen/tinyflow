/*!
 *  Copyright (c) 2016 by Contributors
 * \file base.h
 * \brief defines basic types
 */
#ifndef NNVM_RTC_BASE_H_
#define NNVM_RTC_BASE_H_

#include <dmlc/logging.h>
#include <nnvm/graph.h>
#include <nnvm/node.h>
#include <functional>
#include <vector>
#include <string>

namespace nnvm {
namespace rtc {

using FusionNode    = Node;
using FusionNodePtr = NodePtr;
using FusionGraph   = Graph;
using Kernel = std::pair<std::string, std::string>;

class AST;
using ASTPtr = std::shared_ptr<AST>; // TODO consider unique_ptr?

using FCodeGen = std::function<std::vector<ASTPtr>(
    const NodePtr& nodeptr,
    const std::vector<ASTPtr>& inputs)>;


} // namespace rtc
} // namespace nnvm

#endif  // NNVM_RTC_BASE_H_
