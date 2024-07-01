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

#include "ExtraTensorGenerator.h"

#include "ops/BackPropAccumulator.h"
#include "ops/BinaryArithmeticLayer.h"
#include "ops/ConvolutionLayer.h"
#include "ops/DepthwiseConvolutionLayer.h"
#include "ops/ElementwiseActivationLayer.h"
#include "ops/FullyConnectedLayer.h"
#include "ops/LossMeanSquaredErrorLayer.h"
#include "ops/LossCategoricalCrossentropyLayer.h"
#include "ops/MeanLayer.h"
#include "ops/GradientApplier.h"
#include "ops/PadLayer.h"
#include "ops/PoolLayer.h"
#include "ops/ReshapeLayer.h"
#include "ops/SoftMaxLayer.h"

namespace onert
{
namespace backend
{
namespace train
{

ExtraTensorGenerator::ExtraTensorGenerator(const ir::train::TrainableGraph &tgraph,
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

void ExtraTensorGenerator::handle_requests(const ir::OperationIndex &op_idx,
                                           const ExtraTensorRequests &requests)
{
  // use 'auto i' induced to 'integer' -> this causes comparsion error.
  for (size_t i = 0; i < requests.size(); i++)
  {
    auto r = requests[i];

    ExtraTensorIndex idx(op_idx, i);
    _tensor_builder->registerExtraTensorInfo(idx, r.info, r.layout);
    _tensor_builder->notifyFirstUse(idx);
  }

  // if dynamic, release before configure next nodes
  for (size_t i = 0; i < requests.size(); i++)
  {
    if (requests[i].lifetime == ExtraTensorLifeTime::DYNAMIC)
    {
      ExtraTensorIndex idx(op_idx, i);
      _tensor_builder->notifyFirstUse(idx);
    }
  }
  return;
}

void ExtraTensorGenerator::visit(const ir::train::operation::FullyConnected &node)
{
  using ir::train::operation::FullyConnected;

  const auto in_index{node.getInputs().at(FullyConnected::Input::INPUT)};
  const auto weights_index{node.getInputs().at(FullyConnected::Input::WEIGHT)};

  auto in_tensor = _tensor_reg->getPortableTensor(in_index);
  auto weights_tensor = _tensor_reg->getTrainableTensor(weights_index);

  auto requests = ops::FullyConnectedLayer::requestExtraTensors(weights_tensor, in_tensor);

  auto op_idx = _node_to_idx[&node];
  handle_requests(op_idx, requests);
  return;
}

} // namespace train
} // namespace backend
} // namespace onert
