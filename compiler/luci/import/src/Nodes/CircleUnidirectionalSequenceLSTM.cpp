/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "luci/Import/Nodes/CircleUnidirectionalSequenceLSTM.h"

#include <luci/IR/Nodes/CircleUnidirectionalSequenceLSTM.h>

#include <loco.h>

namespace luci
{

bool CircleUnidirectionalSequenceLSTMGraphBuilder::validate(const ValidateArgs &args) const
{
  return GraphBuilder::validate(args, 24);
}

CircleNode *CircleUnidirectionalSequenceLSTMGraphBuilder::build_node(
  const circle::OperatorT &op, const std::vector<CircleNode *> &inputs, loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleUnidirectionalSequenceLSTM>();
  node->input(inputs.at(0));
  node->input_to_input_weights(inputs.at(1)); // Optional
  node->input_to_forget_weights(inputs.at(2));
  node->input_to_cell_weights(inputs.at(3));
  node->input_to_output_weights(inputs.at(4));

  node->recurrent_to_input_weights(inputs.at(5)); // Optional
  node->recurrent_to_forget_weights(inputs.at(6));
  node->recurrent_to_cell_weights(inputs.at(7));
  node->recurrent_to_output_weights(inputs.at(8));

  node->cell_to_input_weights(inputs.at(9));   // Optional
  node->cell_to_forget_weights(inputs.at(10)); // Optional
  node->cell_to_output_weights(inputs.at(11)); // Optional

  node->input_gate_bias(inputs.at(12)); // Optional
  node->forget_gate_bias(inputs.at(13));
  node->cell_gate_bias(inputs.at(14));
  node->output_gate_bias(inputs.at(15));

  node->projection_weights(inputs.at(16)); // Optional
  node->projection_bias(inputs.at(17));    // Optional

  node->output_state(inputs.at(18));
  node->cell_state(inputs.at(19));

  node->input_layer_norm_coefficients(inputs.at(20));  // Optional
  node->forget_layer_norm_coefficients(inputs.at(21)); // Optional
  node->cell_layer_norm_coefficients(inputs.at(22));   // Optional
  node->output_layer_norm_coefficients(inputs.at(23)); // Optional

  const auto *options = op.builtin_options.AsUnidirectionalSequenceLSTMOptions();
  node->fusedActivationFunction(luci_actfunc(options->fused_activation_function));
  node->cell_clip(options->cell_clip);
  node->proj_clip(options->proj_clip);
  node->time_major(options->time_major);
  node->asymmetric_quantize_inputs(options->asymmetric_quantize_inputs);

  return node;
}

} // namespace luci
