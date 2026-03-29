#pragma once

// Defines the SwiGLU layer that combines gate and value activations.

#include <type_traits>

#include "Layer.h"
#include "celeritas/kernels/KernelFactory.h"

namespace eCEL {

template <typename T>
class SwigluLayer : public Layer<T> {
public:
    using Layer<T>::forward;

    explicit SwigluLayer(eUTIL::DeviceType device)
        : Layer<T>(device) {}

    void forward(const ForwardContext& ctx,
                 TensorListView<T> inputs,
                 eUTIL::Tensor<T>& output) override;
};

}  // namespace eCEL
