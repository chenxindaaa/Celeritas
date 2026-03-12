#ifndef ADD_CU_H
#define ADD_CU_H
#include "baseutil/tensor/tensor.h"

namespace kernel {
void add_kernel_cu(const baseutil::tensor::Tensor<int>& input1,
                   const baseutil::tensor::Tensor<int>& input2,
                   baseutil::tensor::Tensor<int>& output, void* stream = nullptr);
}  // namespace kernel
#endif  // ADD_CU_H
