#pragma once

#include <type_traits>
#include <utility>

#include "Layer.h"
#include "celeritas/kernels/KernelFactory.h"

namespace eCEL {

template <typename Tvalue>
struct AddParams {
    Parameter<Tvalue> value;

    explicit AddParams(Parameter<Tvalue> value_param)
        : value(std::move(value_param)) {}
};

template <typename T>
class AddLayer : public Layer {
public:
    static_assert(std::is_same_v<T, float>,
                  "AddLayer currently supports float tensors only");

    using ParamType = AddParams<T>;

    explicit AddLayer(eUTIL::DeviceType device,
                      ParamType params)
        : Layer(device),
          m_params(std::move(params)) {}

    explicit AddLayer(eUTIL::DeviceType device)
        : Layer(device),
          m_params(Parameter<T>(eUTIL::Tensor<T>(device), "value")) {}

    explicit AddLayer(eUTIL::DeviceType device,
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
