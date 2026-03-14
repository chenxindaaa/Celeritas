#pragma once

#include <cstdint>

#include "baseutil/tensor/tensor.h"

namespace eCEL {

void embKernelCpu(const eUTIL::Tensor<int>& input,
                  const eUTIL::Tensor<float>& weight,
                  eUTIL::Tensor<float>& output,
                  int32_t vocabSize,
                  void* stream = nullptr);

}  // namespace eCEL
