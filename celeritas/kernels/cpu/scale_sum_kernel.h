#ifndef SCALE_SUM_KERNEL_H
#define SCALE_SUM_KERNEL_H
#include <core/Tensor.h>
namespace kernel {
void scale_sum_kernel_cpu(const tensor::Tensor& value, const tensor::Tensor& scale, 
                          const tensor::Tensor& output, int t, int d, int stride,
                          void* stream = nullptr);
}
#endif  // SCALE_KERNEL_H

