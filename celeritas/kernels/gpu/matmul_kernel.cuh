#pragma once

#include "udm/common/cudaConfig.h"
#include "udm/core/Tensor.h"

namespace eCEL {

template<typename T>
void matmulKernelCu(const eUTIL::Tensor<T>& input, const eUTIL::Tensor<T>& weight,
                    eUTIL::Tensor<T>& output, const float scale = 1.f,
                    const eUTIL::CudaConfig* config = nullptr)
{
    (void)input;
    (void)weight;
    (void)output;
    (void)scale;
    (void)config;
    return;
}

template<>
void matmulKernelCu(const eUTIL::Tensor<float>& input, const eUTIL::Tensor<float>& weight,
                    eUTIL::Tensor<float>& output, const float scale,
                    const eUTIL::CudaConfig* config);

}  // namespace eCEL

