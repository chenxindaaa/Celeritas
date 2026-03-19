#pragma once

#include <cstdint>

#include "udm/core/Tensor.h"

namespace eCEL {

template<typename T>
void embKernelCu(const eUTIL::Tensor<T>& input,
                 const eUTIL::Tensor<T>& weight,
                 eUTIL::Tensor<T>& output,
                 int32_t vocabSize,
                 void* stream = nullptr)
{
    (void)input;
    (void)weight;
    (void)output;
    (void)vocabSize;
    (void)stream;
    return;
}

template<>
void embKernelCu(const eUTIL::Tensor<float>& input,
                 const eUTIL::Tensor<float>& weight,
                 eUTIL::Tensor<float>& output,
                 int32_t vocabSize,
                 void* stream);

}  // namespace eCEL
