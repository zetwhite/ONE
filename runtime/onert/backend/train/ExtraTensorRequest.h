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

#ifndef __ONERT_BACKEND_EXTRA_TENSOR_REQUEST_H__
#define __ONERT_BACKEND_EXTRA_TENSOR_REQUEST_H__

#include "ir/OperandInfo.h"
#include "backend/IPortableTensor.h"

namespace onert
{
namespace backend
{
namespace train
{

enum class ExtraTensorLifeTime
{
  // TODO: find better way to explain lifetime
  DYNAMIC, // << exist during forward() or backward()
  STATIC,  // << always exists
};

struct ExtraTensorRequest
{
  static ExtraTensorRequest createRequestLike(const IPortableTensor *origin)
  {
    ExtraTensorRequest r = {.info = origin->get_info(),
                            .layout = origin->layout(),
                            .lifetime = ExtraTensorLifeTime::DYNAMIC};
    return r;
  }

  ir::OperandInfo info;
  ir::Layout layout;
  ExtraTensorLifeTime lifetime;
};

using ExtraTensorRequests = std::vector<ExtraTensorRequest>;

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_EXTRA_TENSOR_REQUEST_H__