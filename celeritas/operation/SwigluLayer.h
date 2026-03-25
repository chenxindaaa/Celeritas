#pragma once

#include <type_traits>
#include <utility>

#include "Layer.h"
#include "celeritas/kernels/KernelFactory.h"

namespace eCEL {

template <typename Tvalue>
struct SwigluParams {
    Parameter<Tvalue> value;

    explicit SwigluParams(Parameter<Tvalue> value_param)
        : value(std::move(value_param)) {}
};

template <typename T>
class SwigluLayer : public Layer {
public:
    static_assert(std::is_same_v<T, float>,
                  "SwigluLayer currently supports float tensors only");

    using ParamType = SwigluParams<T>;

    explicit SwigluLayer(eUTIL::DeviceType device,
                         ParamType params)
        : Layer(device),
          m_params(std::move(params)) {}

    explicit SwigluLayer(eUTIL::DeviceType device,
                         Parameter<T> value)
        : Layer(device),
          m_params(std::move(value)) {}

    void forward(const ForwardContext& ctx,
                 const Tensor& input,
                 Tensor& output) override;

    const ParamType& params() const { return m_params; }
    ParamType& params() { return m_params; }

private:
    ParamType m_params;
};

}  // namespace eCEL
