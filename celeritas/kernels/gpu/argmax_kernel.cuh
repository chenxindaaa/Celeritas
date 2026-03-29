#pragma once

#include <cstddef>

#include "udm/core/Tensor.h"

namespace eCEL {

void argmaxKernelCu(const eUTIL::Tensor<float>& input,
                    eUTIL::Tensor<std::size_t>& output,
                    void* stream = nullptr);

}
