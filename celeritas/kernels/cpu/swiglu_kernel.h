#pragma once

#include "udm/core/Tensor.h"
namespace eCEL {

template<typename T>
void swigluKernelCpu(const eUTIL::Tensor<T>& input1, const eUTIL::Tensor<T>& input2,
                    eUTIL::Tensor<T>& output, void* stream) {
    (void)stream;
    // CHECK_EQ(input1.is_empty(), false);
    // CHECK_EQ(input2.is_empty(), false);
    // CHECK_EQ(output.is_empty(), false);

    // CHECK(input1.device_type() == base::DeviceType::kDeviceCPU);
    // CHECK(input2.device_type() == base::DeviceType::kDeviceCPU);
    // CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

    arma::Col<T> input1_vec(const_cast<T*>(input1.data()), input1.size(), false, true);
    arma::Col<T> input2_vec(const_cast<T*>(input2.data()), input2.size(), false, true);
    arma::Col<T> output_vec(output.data(), output.size(), false, true);

    arma::Col<T> input1_vec_new = (1.0f / (1.0f + arma::exp(-input1_vec)));
    output_vec = input1_vec % input1_vec_new % input2_vec;
}
}  // namespace eCEL