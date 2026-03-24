#pragma once

#include <cstdint>

#include "udm/common/cudaConfig.h"
#include "udm/core/Tensor.h"

namespace eCEL {

template <typename Tin, typename Tw = Tin, typename Ts = float, typename Tout = Tin>
void matmulKernelCu(const eUTIL::Tensor<Tin>& input, const eUTIL::Tensor<Tw>& weight,
                    const eUTIL::Tensor<Ts>& scaler, eUTIL::Tensor<Tout>& output,
                    int32_t group_size = 1,
                    const eUTIL::CudaConfig* config = nullptr)
{
    (void)input;
    (void)weight;
    (void)scaler;
    (void)output;
    (void)group_size;
    (void)config;
    return;
}

template<>
void matmulKernelCu(const eUTIL::Tensor<float>& input, const eUTIL::Tensor<float>& weight,
                    const eUTIL::Tensor<float>& scaler, eUTIL::Tensor<float>& output,
                    int32_t group_size,
                    const eUTIL::CudaConfig* config);

template<>
void matmulKernelCu(const eUTIL::Tensor<float>& input, const eUTIL::Tensor<int8_t>& weight,
                    const eUTIL::Tensor<float>& scaler, eUTIL::Tensor<float>& output,
                    int32_t group_size,
                    const eUTIL::CudaConfig* config);

}  // namespace eCEL
