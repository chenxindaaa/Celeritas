#pragma once
#include <armadillo>

#include "udm/core/Tensor.h"

namespace eCEL {

template<typename T>
void add_kernel_cpu(const eUTIL::Tensor<T>& input1, const eUTIL::Tensor<T>& input2,
                    eUTIL::Tensor<T>& output, void* stream = nullptr)
{
    (void)stream;
    // UNUSED(stream);
    // CHECK_EQ(input1.empty(), false);
    // CHECK_EQ(input2.empty(), false);
    // CHECK_EQ(output.empty(), false);

    // CHECK_EQ(input1.size(), input2.size());
    // CHECK_EQ(input1.size(), output.size());

    arma::Col<T> input_vec1(const_cast<T*>(input1.data()), input1.size(), false, true);
    arma::Col<T> input_vec2(const_cast<T*>(input2.data()), input2.size(), false, true);
    arma::Col<T> output_vec(output.data(), output.size(), false, true);
    output_vec = input_vec1 + input_vec2;
}

}  // namespace eCEL