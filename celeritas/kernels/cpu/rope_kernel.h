#pragma once

// Defines the CPU RoPE kernel that rotates query and key tensors in place.

#include <cstdint>
#include <stdexcept>

#include "udm/core/Tensor.h"

namespace eCEL {

template <typename T>
void ropeKernelCpu(int32_t pos,
                   int32_t dim,
                   int32_t kvDim,
                   int32_t headSize,
                   eUTIL::Tensor<T>& inputQ,
                   eUTIL::Tensor<T>& inputK,
                   const eUTIL::Tensor<T>& sinCache,
                   const eUTIL::Tensor<T>& cosCache,
                   void* stream = nullptr) {
    (void)stream;

    if (pos < 0 || dim < 0 || kvDim < 0 || headSize <= 0) {
        throw std::invalid_argument("rope kernel expects non-negative dims and positive head size");
    }
    if ((dim % 2) != 0 || (kvDim % 2) != 0 || (headSize % 2) != 0) {
        throw std::invalid_argument("rope kernel requires dim, kvDim, and headSize to be even");
    }
    if (inputQ.size() < static_cast<std::size_t>(dim) ||
        inputK.size() < static_cast<std::size_t>(kvDim)) {
        throw std::invalid_argument("rope kernel input tensor size does not match configured dims");
    }
    if (sinCache.dimSize() != 2 || cosCache.dimSize() != 2) {
        throw std::invalid_argument("rope cache tensors must be 2D");
    }
    if (sinCache.getDim(0) != cosCache.getDim(0) || sinCache.getDim(1) != cosCache.getDim(1)) {
        throw std::invalid_argument("rope cache tensor shapes must match");
    }
    if (sinCache.getDim(1) != headSize) {
        throw std::invalid_argument("rope cache head size does not match kernel head size");
    }
    if (pos >= sinCache.getDim(0)) {
        throw std::out_of_range("rope position exceeds cache sequence length");
    }

    for (int32_t i = 0; i < dim; i += 2) {
        const int32_t headDim = i % headSize;
        const T fci = *(sinCache.data() + pos * headSize + headDim);
        const T fcr = *(cosCache.data() + pos * headSize + headDim);

        T* queryVec = inputQ.data();
        const T q0 = queryVec[i];
        const T q1 = queryVec[i + 1];
        queryVec[i] = q0 * fcr - q1 * fci;
        queryVec[i + 1] = q0 * fci + q1 * fcr;

        if (i >= kvDim) {
            continue;
        }

        T* keyVec = inputK.data();
        const T k0 = keyVec[i];
        const T k1 = keyVec[i + 1];
        keyVec[i] = k0 * fcr - k1 * fci;
        keyVec[i + 1] = k0 * fci + k1 * fcr;
    }
}

}  // namespace eCEL
