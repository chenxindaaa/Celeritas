#ifndef LLAMA_INFER_SOFTMAX_KERNEL_H
#define LLAMA_INFER_SOFTMAX_KERNEL_H
#include "core/Tensor.h"
namespace kernel {
void softmax_inplace_cpu(const tensor::Tensor& input, void* stream = nullptr);
}  // namespace kernel
#endif  // LLAMA_INFER_SOFTMAX_KERNEL_H

