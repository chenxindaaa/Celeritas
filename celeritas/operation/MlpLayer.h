#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include "AddLayer.h"
#include "MatmulLayer.h"
#include "SwigluLayer.h"

namespace eCEL {

template <typename Tweight>
struct MlpParams {
    MatmulParams<Tweight> gate_proj;
    MatmulParams<Tweight> down_proj;
    MatmulParams<Tweight> up_proj;

    MlpParams(MatmulParams<Tweight> gate_proj_param,
              MatmulParams<Tweight> down_proj_param,
              MatmulParams<Tweight> up_proj_param)
        : gate_proj(std::move(gate_proj_param)),
          down_proj(std::move(down_proj_param)),
          up_proj(std::move(up_proj_param)) {}
};

template <typename Tact, typename Tweight = Tact>
class MlpLayer : public Layer {
public:
    static_assert(std::is_same_v<Tact, float>,
                  "MlpLayer currently supports float activation tensors only");

    using ParamType = MlpParams<Tweight>;

    explicit MlpLayer(eUTIL::DeviceType device,
                      ParamType params)
        : Layer(device),
          m_params(std::move(params)),
          m_gateLayer(device, m_params.gate_proj),
          m_upLayer(device, m_params.up_proj),
          m_downLayer(device, m_params.down_proj),
          m_swigluLayer(device, Parameter<float>(Tensor(device), "value")),
          m_addLayer(device, Parameter<float>(Tensor(device), "value")) {}

    void forward(const ForwardContext& ctx,
                 const Tensor& input,
                 Tensor& output) override;

    const ParamType& params() const { return m_params; }
    ParamType& params() { return m_params; }

private:
    static Tensor makeMatmulOutputTensor(const Tensor& input,
                                         const eUTIL::Tensor<Tweight>& weight,
                                         eUTIL::DeviceType device);

    ParamType m_params;
    MatmulLayer<Tact, Tweight> m_gateLayer;
    MatmulLayer<Tact, Tweight> m_upLayer;
    MatmulLayer<Tact, Tweight> m_downLayer;
    SwigluLayer<float> m_swigluLayer;
    AddLayer<float> m_addLayer;
};

}  // namespace eCEL
