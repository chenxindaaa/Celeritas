#pragma once
#include <udm/core/Tensor.h>
namespace eCEL {
template <typename T>
void swigluKernelCu(const eUTIL::Tensor<T>& input1, const eUTIL::Tensor<T>& input2,
                    eUTIL::Tensor<T>& output, void* stream);

template <>
void swigluKernelCu(const eUTIL::Tensor<float>& input1, const eUTIL::Tensor<float>& input2,
                    eUTIL::Tensor<float>& output, void* stream);
}  // namespace eCEL