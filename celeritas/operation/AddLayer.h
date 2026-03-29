#pragma once

// Defines the residual add layer that sums two activation tensors.

#include "Layer.h"
#include "celeritas/kernels/KernelFactory.h"

namespace eCEL {

template <typename T>
class AddLayer : public Layer<T> {
public:
    using Layer<T>::forward;

    explicit AddLayer(eUTIL::DeviceType device)
        : Layer<T>(device) {}

    void forward(const ForwardContext& ctx,
                 TensorListView<T> inputs,
                 eUTIL::Tensor<T>& output) override;
};

}  // namespace eCEL
