#pragma once
#include <armadillo>

#include "udm/common/cudaConfig.h"
#include "udm/core/Tensor.h"

namespace eCEL {

template<typename Tin, typename Tw = Tin, typename Ts = float, typename Tout = Tin>
void matmulKernelCpu(const eUTIL::Tensor<Tin>& input, const eUTIL::Tensor<Tw>& weight,
                     const eUTIL::Tensor<Ts>& scaler, eUTIL::Tensor<Tout>& output,
                     int32_t group_size = 1, const eUTIL::CudaConfig* config = nullptr) {
    (void)config;
    (void)group_size;

    const float scale = scaler.empty() ? 1.f : static_cast<float>(scaler.data()[0]);

    const Tin* input_ptr = input.data();
    const Tw* weight_ptr = weight.data();
    Tout* output_ptr = output.data();

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
    arma::Mat<Tin> input_mat(const_cast<Tin*>(input_ptr), in_dim1, in_dim0, false, true);
    arma::Mat<Tw> weight_mat(const_cast<Tw*>(weight_ptr), wei_dim1, wei_dim0, false, true);
    arma::Mat<Tout> output_mat(output_ptr, in_dim1, wei_dim0, false, true);
    output_mat = ((input_mat * weight_mat)) * scale;
}

}  // namespace eCEL
