#include <armadillo>

#include "udm/core/Tensor.h"
namespace eCEL {
// scale @ (v0,    
//          v1,  = scale[0] * v0 + scale[1] * v + scale[2] * v2
//          v2)
template <typename T>
void scalesumKernelCpu(const eUTIL::Tensor<T>& value, const eUTIL::Tensor<T>& scale,
                       eUTIL::Tensor<T>& output, int pos, int size, int stride,
                       void* stream = nullptr) {
    (void)stream;
    // CHECK_EQ(value.is_empty(), false);
    // CHECK_EQ(scale.is_empty(), false);
    // CHECK_EQ(output.is_empty(), false);
    // CHECK_EQ(size, value.size());
    // CHECK_EQ(size, output.size());
    arma::Col<T> scale_vec(const_cast<float*>(scale.data()), scale.size(), false, true);
    arma::Col<T> output_vec(output.data(), output.size(), false, true);

    for (int i = 0; i <= pos; ++i) {
        arma::Col<T> value_vec(const_cast<float*>(value.data()) + i * stride, value.size(), false,
                               true);
        output_vec += scale_vec[i] * value_vec;
    }
}
}  // namespace eCEL