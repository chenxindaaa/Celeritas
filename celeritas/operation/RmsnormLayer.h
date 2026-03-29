#pragma once

// Defines the RMSNorm layer for activation normalization.

#include <type_traits>
#include <utility>

#include "Layer.h"
#include "celeritas/kernels/KernelFactory.h"

namespace eCEL {

template <typename T>
struct RmsnormParams {
    Parameter<T> weight;

    RmsnormParams() = default;

    explicit RmsnormParams(Parameter<T> weight_param)
        : weight(std::move(weight_param)) {}
};

template <typename T>
class RmsnormLayer : public Layer<T> {
public:
    using ParamType = RmsnormParams<T>;
    using Layer<T>::forward;

    explicit RmsnormLayer(eUTIL::DeviceType device,
                          ParamType params)
        : Layer<T>(device),
          m_params(std::move(params)) {}

    explicit RmsnormLayer(eUTIL::DeviceType device = eUTIL::DeviceType::kCpu)
        : Layer<T>(device),
          m_params() {}

    explicit RmsnormLayer(eUTIL::DeviceType device,
                          Parameter<T> weight)
        : Layer<T>(device),
          m_params(std::move(weight)) {}

    void forward(const ForwardContext& ctx,
                 TensorListView<T> inputs,
                 eUTIL::Tensor<T>& output) override;

    void to(eUTIL::DeviceType device) override {
        this->m_device = device;
        m_params.weight.to(device);
    }

    void setParams(ParamType params) { m_params = std::move(params); }
    void setWeight(Parameter<T> weight) { m_params.weight = std::move(weight); }

    const ParamType& params() const { return m_params; }
    ParamType& params() { return m_params; }

private:
    ParamType m_params;
};

}  // namespace eCEL
