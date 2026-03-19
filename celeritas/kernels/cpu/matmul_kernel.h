#pragma once
#include <armadillo>

#include "udm/common/cudaConfig.h"
#include "udm/core/Tensor.h"

namespace eCEL {

template<typename T>
void matmulKernelCpu(const eUTIL::Tensor<T>& input, const eUTIL::Tensor<T>& weight,
                     eUTIL::Tensor<T>& output, const float scale = 1.f,
                     const eUTIL::CudaConfig* config = nullptr) {
    (void)config;
    // CHECK(input.is_empty() == false);
    // CHECK(weight.is_empty() == false);
    // CHECK(output.is_empty() == false);
    // CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
    // CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
    // CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

    const T* input_ptr = input.data();
    const T* weight_ptr = weight.data();
    T* output_ptr = output.data();

    int32_t in_dim1 = 1;
    int32_t in_dim0 = 1;
    if (input.dimSize() == 2) {
        in_dim0 = input.getDim(0);
        in_dim1 = input.getDim(1);
    } else if (input.dimSize() == 1) {
        in_dim0 = input.getDim(0);
    } else {
        assert(false);
        // LOG(FATAL) << "The input tensor has a wrong dim size.";
    }

    // CHECK_EQ(weight.dims_size(), 2);
    const int32_t wei_dim0 = weight.getDim(0);
    const int32_t wei_dim1 = weight.getDim(1);
    // CHECK_EQ(in_dim0, wei_dim1);

    // CHECK_EQ(output.size(), wei_dim0 * in_dim1);
    arma::Mat<T> input_mat(const_cast<T*>(input_ptr), in_dim1, in_dim0, false, true);
    arma::Mat<T> weight_mat(const_cast<T*>(weight_ptr), wei_dim1, wei_dim0, false, true);
    arma::Mat<T> output_mat(output_ptr, in_dim1, wei_dim0, false, true);
    output_mat = ((input_mat * weight_mat)) * scale;
}

}  // namespace eCEL
