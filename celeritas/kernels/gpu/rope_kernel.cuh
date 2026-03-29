#pragma once

#include "udm/core/Tensor.h"

namespace eCEL {

template <typename T>
void ropeKernelCu(int32_t pos,
                  int32_t dim,
                  int32_t kvDim,
                  int32_t headSize,
                  eUTIL::Tensor<T>& inputQ,
                  eUTIL::Tensor<T>& inputK,
                  const eUTIL::Tensor<T>& sinCache,
                  const eUTIL::Tensor<T>& cosCache,
                  void* stream = nullptr) {
    return;
}

template <>
void ropeKernelCu(int32_t pos,
                  int32_t dim,
                  int32_t kvDim,
                  int32_t headSize,
                  eUTIL::Tensor<float>& inputQ,
                  eUTIL::Tensor<float>& inputK,
                  const eUTIL::Tensor<float>& sinCache,
                  const eUTIL::Tensor<float>& cosCache,
                  void* stream);

}  // namespace eCEL

