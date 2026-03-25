#pragma once

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include "Layer.h"
#include "celeritas/kernels/KernelFactory.h"

namespace eCEL {

template <typename Tweight>
struct MatmulParams {
    Parameter<Tweight> weight;
    Parameter<float> scaler;
    int32_t group_size = 1;

    MatmulParams(Parameter<Tweight> weight_param,
                 Parameter<float> scaler_param,
                 int32_t group_size_param = 1)
        : weight(std::move(weight_param)),
          scaler(std::move(scaler_param)),
          group_size(group_size_param) {}
};

template <typename Tact, typename Tweight = Tact>
class MatmulLayer : public Layer {
public:
    using ParamType = MatmulParams<Tweight>;

    explicit MatmulLayer(eUTIL::DeviceType device,
                         ParamType params)
        : Layer(device),
          m_params(std::move(params)) {}

    explicit MatmulLayer(eUTIL::DeviceType device,
                         Parameter<Tweight> weight,
                         Parameter<float>scaler,
                         int32_t group_size = 1)
        : Layer(device),
          m_params(std::move(weight), std::move(scaler), group_size) {}

    void forward(const ForwardContext& ctx,
                 const Tensor& input,
                 Tensor& output) override;

    const ParamType& params() const { return m_params; }
    ParamType& params() { return m_params; }

private:
    ParamType m_params;
};

}  // namespace eCEL
