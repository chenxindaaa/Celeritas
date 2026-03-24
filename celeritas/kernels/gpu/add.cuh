#pragma once

#include "udm/core/Tensor.h"

namespace eCEL {

template<typename T>
void add_kernel_cu(const eUTIL::Tensor<T>& input1,
                   const eUTIL::Tensor<T>& input2,
                   eUTIL::Tensor<T>& output, void* stream = nullptr)
{
    return;
}

template<>
void add_kernel_cu(const eUTIL::Tensor<float>& input1,
                   const eUTIL::Tensor<float>& input2,
                   eUTIL::Tensor<float>& output, void* stream);

template<>
void add_kernel_cu(const eUTIL::Tensor<int8_t>& input1,
                   const eUTIL::Tensor<int8_t>& input2,
                   eUTIL::Tensor<int8_t>& output, void* stream);
}  // namespace eCEL
