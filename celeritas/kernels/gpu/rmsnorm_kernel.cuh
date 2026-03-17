#pragma once

#include "udm/core/Tensor.h"

namespace eCEL {


template<typename T>
void rmsKernelCu(const eUTIL::Tensor<T>& input, const eUTIL::Tensor<T>& weight,
                 eUTIL::Tensor<T>& output, void* stream = nullptr)
{
    return;
}

template<>
void rmsKernelCu(const eUTIL::Tensor<float>& input, const eUTIL::Tensor<float>& weight,
                 eUTIL::Tensor<float>& output, void* stream);

// template<typename T>
// void rmsKernelCuDim(const eUTIL::Tensor<T>& input, const eUTIL::Tensor<T>& weight,
//                     eUTIL::Tensor<T>& output, int32_t dim, void* stream = nullptr)
// {
//     return false;
// }
}  // namespace eCEL