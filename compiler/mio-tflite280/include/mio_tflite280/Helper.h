/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MIO_TFLITE280_HELPER_H__
#define __MIO_TFLITE280_HELPER_H__

#include <mio/tflite/schema_generated.h>

namespace mio
{
namespace tflite
{

::tflite::BuiltinOperator builtin_code_neutral(const ::tflite::OperatorCode *opcode);
bool is_valid(const ::tflite::OperatorCode *opcode);
bool is_custom(const ::tflite::OperatorCode *opcode);
std::string opcode_name(const ::tflite::OperatorCode *opcode);
const char *tensor_type(const ::tflite::Tensor *tensor);
const char *tensor_name(const ::tflite::Tensor *tensor);

} // namespace tflite
} // namespace mio

#endif // __MIO_TFLITE280_HELPER_H__
