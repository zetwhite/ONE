/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_EXTRA_TENSOR_ALLOCATOR_H__
#define __ONERT_BACKEND_EXTRA_TENSOR_ALLOCATOR_H__

// to construct vsitior
#include "ir/train/TrainableOperationVisitor.h"
#include "ir/train/Operations.Include.h"

// ...
#include "ir/train/TrainableGraph.h"
#include "ir/train/ITrainableOperation.h"
#include "ir/Index.h"
#include "TensorBuilder.h"

namespace onert
{
namespace backend
{
namespace train
{

using ExtraTensor = Tensor;

enum class ExtraTensorLifeTime
{
  // TODO: find better way to explain lifetime
  DYNAMIC, // << exist during forward() or backward()
  STATIC,  // << always exists
};

struct ExtraTensorRequest
{
  ir::OperandInfo info;
  ir::Layout layout;
  ExtraTensorLifeTime lifetime;
};

using ExtraTensorRequests = std::vector<ExtraTensorRequest>;

class ExtraTensorAllocator : public ir::train::TrainableOperationVisitor
{
public:
  ExtraTensorAllocator() = delete;
  ExtraTensorAllocator(const ir::train::TrainableGraph &tgraph,
                       std::shared_ptr<TensorBuilder> &tensor_builder,
                       std::shared_ptr<ITensorRegistry> &tensor_registry)
    : _tgraph(tgraph), _tensor_builder(tensor_builder)
  {
    _tensor_reg = std::dynamic_pointer_cast<TensorRegistry>(tensor_registry);

    for (const auto &index : _tgraph.topolSortOperations())
    {
      const auto &node = _tgraph.operation(index);
      _node_to_idx[&node] = index;
    }
  };

public:
  void visit(const ir::train::operation::FullyConnected &node) override;

  // void visit(const train::operation::BinaryArithmetic &node) override;
  // void visit(const train::operation::Conv2D &node) override;
  // void visit(const train::operation::DepthwiseConv2D &node) override;
  // void visit(const train::operation::ElementwiseActivation &node) override;
  // void visit(const train::operation::Loss &node) override;
  // void visit(const train::operation::Pad &node) override;
  // void visit(const train::operation::Pool2D &node) override;
  // void visit(const train::operation::Reduce &node) override;
  // void visit(const train::operation::Reshape &node) override;
  // void visit(const train::operation::Softmax &node) override;

private:
  void handle_requests(const ir::OperationIndex &, const ExtraTensorRequests &);

private:
  const ir::train::TrainableGraph &_tgraph;
  std::shared_ptr<TensorBuilder> _tensor_builder;
  std::shared_ptr<TensorRegistry> _tensor_reg;
  std::unordered_map<const ir::train::ITrainableOperation *, ir::OperationIndex> _node_to_idx;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_EXTRA_TENSOR_ALLOCATOR_H__
