#pragma once

// Defines the feed-forward block that combines projections and SwiGLU without residual add.

#include <cstdint>
#include <type_traits>
#include <utility>

#include "MatmulLayer.h"
#include "SwigluLayer.h"

namespace eCEL {

template <typename Tweight>
struct MlpParams {
    MatmulParams<Tweight> gate_proj;
    MatmulParams<Tweight> down_proj;
    MatmulParams<Tweight> up_proj;

    MlpParams() = default;

    MlpParams(MatmulParams<Tweight> gate_proj_param,
              MatmulParams<Tweight> down_proj_param,
              MatmulParams<Tweight> up_proj_param)
        : gate_proj(std::move(gate_proj_param)),
          down_proj(std::move(down_proj_param)),
          up_proj(std::move(up_proj_param)) {}
};

template <typename Tact, typename Tweight = Tact>
class MlpLayer : public Layer<Tact> {
public:
    static_assert(std::is_same_v<Tact, float>,
                  "MlpLayer currently supports float activation tensors only");

    using ParamType = MlpParams<Tweight>;
    using Layer<Tact>::forward;

    explicit MlpLayer(eUTIL::DeviceType device,
                      ParamType params)
        : Layer<Tact>(device),
          m_params(std::move(params)),
          m_gateLayer(device, m_params.gate_proj),
          m_upLayer(device, m_params.up_proj),
          m_downLayer(device, m_params.down_proj),
          m_swigluLayer(device) {}

    explicit MlpLayer(eUTIL::DeviceType device = eUTIL::DeviceType::kCpu)
        : Layer<Tact>(device),
          m_params(),
          m_gateLayer(device),
          m_upLayer(device),
          m_downLayer(device),
          m_swigluLayer(device) {}

    void forward(const ForwardContext& ctx,
                 TensorListView<Tact> inputs,
                 eUTIL::Tensor<Tact>& output) override;

    void to(eUTIL::DeviceType device) override {
        this->m_device = device;
        m_gateLayer.to(device);
        m_upLayer.to(device);
        m_downLayer.to(device);
        m_swigluLayer.to(device);
        m_params.gate_proj = m_gateLayer.params();
        m_params.up_proj = m_upLayer.params();
        m_params.down_proj = m_downLayer.params();
    }

    void setParams(ParamType params) {
        m_params = std::move(params);
        m_gateLayer.setParams(m_params.gate_proj);
        m_upLayer.setParams(m_params.up_proj);
        m_downLayer.setParams(m_params.down_proj);
    }

    void setGateProjParams(MatmulParams<Tweight> params) {
        m_params.gate_proj = std::move(params);
        m_gateLayer.setParams(m_params.gate_proj);
    }

    void setGateProjWeight(Parameter<Tweight> weight) {
        m_params.gate_proj.weight = std::move(weight);
        m_gateLayer.setWeight(m_params.gate_proj.weight);
    }

    void setDownProjParams(MatmulParams<Tweight> params) {
        m_params.down_proj = std::move(params);
        m_downLayer.setParams(m_params.down_proj);
    }

    void setDownProjWeight(Parameter<Tweight> weight) {
        m_params.down_proj.weight = std::move(weight);
        m_downLayer.setWeight(m_params.down_proj.weight);
    }

    void setUpProjParams(MatmulParams<Tweight> params) {
        m_params.up_proj = std::move(params);
        m_upLayer.setParams(m_params.up_proj);
    }

    void setUpProjWeight(Parameter<Tweight> weight) {
        m_params.up_proj.weight = std::move(weight);
        m_upLayer.setWeight(m_params.up_proj.weight);
    }

    const ParamType& params() const { return m_params; }
    ParamType& params() { return m_params; }

private:
    static eUTIL::Tensor<Tact> makeMatmulOutputTensor(const eUTIL::Tensor<Tact>& input,
                                                      const eUTIL::Tensor<Tweight>& weight,
                                                      eUTIL::DeviceType device);

    ParamType m_params;
    MatmulLayer<Tact, Tweight> m_gateLayer;
    MatmulLayer<Tact, Tweight> m_upLayer;
    MatmulLayer<Tact, Tweight> m_downLayer;
    SwigluLayer<Tact> m_swigluLayer;
};

}  // namespace eCEL
