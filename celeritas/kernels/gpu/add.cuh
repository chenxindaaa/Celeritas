#pragma once

#include "baseutil/tensor/tensor.h"

namespace eCEL {
void add_kernel_cu(const eUTIL::Tensor<int>& input1,
                   const eUTIL::Tensor<int>& input2,
                   eUTIL::Tensor<int>& output, void* stream = nullptr);
}  // namespace eCEL