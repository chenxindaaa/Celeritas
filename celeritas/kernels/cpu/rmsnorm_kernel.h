#pragma once

#include <armadillo>

#include "udm/core/Tensor.h"

namespace eCEL {

template<typename T>
void rmsKernelCpu(const eUTIL::Tensor<T>& input, const eUTIL::Tensor<T>& weight,
                  eUTIL::Tensor<T>& output, void* stream = nullptr)
{
    (void)stream;
    // UNUSED(stream);
    // CHECK(!input.is_empty());
    // CHECK(!weight.is_empty());
    // CHECK(!output.is_empty());

    // CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
    //         weight.device_type() == base::DeviceType::kDeviceCPU &&
    //         output.device_type() == base::DeviceType::kDeviceCPU);

    const int32_t size = static_cast<int32_t>(input.size());

    arma::Col<T> in_tensor(const_cast<T*>(input.data()), size, false, true);
    arma::Col<T> wei_tensor(const_cast<T*>(weight.data()), size, false, true);
    arma::Col<T> out_tensor(output.data(), size, false, true);

    #if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    const float eps = 1e-6f;
    #else
    const float eps = 1e-5f;
    #endif

    const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
    const float rsqrt = 1.f / std::sqrt(mean);
    out_tensor = wei_tensor % (rsqrt * in_tensor);
}

}  // namespace eCEL