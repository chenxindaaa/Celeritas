#pragma once

#include "udm/core/Tensor.h"

namespace eCEL {

template<typename T>
void add_kernel_cu(const eUTIL::Tensor<T>& input1,
                   const eUTIL::Tensor<T>& input2,
                   eUTIL::Tensor<T>& output, void* stream = nullptr);

template<>
void add_kernel_cu(const eUTIL::Tensor<int>& input1,
                   const eUTIL::Tensor<int>& input2,
                   eUTIL::Tensor<int>& output, void* stream);
}  // namespace eCEL

