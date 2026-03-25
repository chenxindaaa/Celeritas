#pragma once

#include <type_traits>
#include <utility>

#include "Layer.h"
#include "celeritas/kernels/KernelFactory.h"

namespace eCEL {

template <typename Tweight>
struct RmsnormParams {
    Parameter<Tweight> weight;

    explicit RmsnormParams(Parameter<Tweight> weight_param)
        : weight(std::move(weight_param)) {}
};

template <typename Tact, typename Tweight = Tact>
class RmsnormLayer : public Layer {
public:
    static_assert(std::is_same_v<Tact, float>,
                  "RmsnormLayer currently supports float activation tensors only");
    static_assert(std::is_same_v<Tweight, float>,
                  "RmsnormLayer currently supports float weight tensors only");

    using ParamType = RmsnormParams<Tweight>;

    explicit RmsnormLayer(eUTIL::DeviceType device,
                          ParamType params)
        : Layer(device),
          m_params(std::move(params)) {}

    explicit RmsnormLayer(eUTIL::DeviceType device,
                          Parameter<Tweight> weight)
        : Layer(device),
          m_params(std::move(weight)) {}

    void forward(const ForwardContext& ctx,
                 const Tensor& input,
                 Tensor& output) override;

    const ParamType& params() const { return m_params; }
    ParamType& params() { return m_params; }

private:
    ParamType m_params;
};

}  // namespace eCEL
