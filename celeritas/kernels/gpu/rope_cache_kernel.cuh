#pragma once

// Defines the CUDA kernel wrapper that precomputes RoPE sin/cos cache tensors.

#include "udm/core/Tensor.h"

namespace eCEL {

template <typename T>
void ropeCacheKernelCu(int32_t headSize,
                       int32_t maxSeqLen,
                       float theta,
                       eUTIL::Tensor<T>& sinCache,
                       eUTIL::Tensor<T>& cosCache,
                       void* stream = nullptr) {
    return;
}

template <>
void ropeCacheKernelCu(int32_t headSize,
                       int32_t maxSeqLen,
                       float theta,
                       eUTIL::Tensor<float>& sinCache,
                       eUTIL::Tensor<float>& cosCache,
                       void* stream);

}  // namespace eCEL
