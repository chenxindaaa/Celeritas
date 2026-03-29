#pragma once

// Defines the CPU kernel that precomputes RoPE sin/cos cache tensors.

#include <cmath>
#include <cstdint>

#include "udm/core/Tensor.h"

namespace eCEL {

template <typename T>
void ropeCacheKernelCpu(int32_t headSize,
                        int32_t maxSeqLen,
                        float theta,
                        eUTIL::Tensor<T>& sinCache,
                        eUTIL::Tensor<T>& cosCache,
                        void* stream = nullptr) {
    (void)stream;

    for (int32_t pos = 0; pos < maxSeqLen; ++pos) {
        for (int32_t headDim = 0; headDim < headSize; ++headDim) {
            const float freq =
                1.0f / std::pow(theta, static_cast<float>(headDim) / static_cast<float>(headSize));
            const float angle = static_cast<float>(pos) * freq;
            *(sinCache.data() + pos * headSize + headDim) = static_cast<T>(std::sin(angle));
            *(cosCache.data() + pos * headSize + headDim) = static_cast<T>(std::cos(angle));
        }
    }
}

}  // namespace eCEL
