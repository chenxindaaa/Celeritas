#pragma once

// Defines the matrix multiplication layer used by linear projections.

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include "Layer.h"
#include "celeritas/kernels/KernelFactory.h"

namespace eCEL {

inline Parameter<float> makeDefaultMatmulScaler() {
    eUTIL::Tensor<float> scaler(eUTIL::DeviceType::kCpu, 1);
    scaler[0] = 1.0f;
    return Parameter<float>(std::move(scaler));
}

template <typename Tweight>
struct MatmulParams {
    Parameter<Tweight> weight;
    Parameter<float> scaler = makeDefaultMatmulScaler();

    MatmulParams() = default;

    explicit MatmulParams(Parameter<Tweight> weight_param)
        : weight(std::move(weight_param)),
          scaler(makeDefaultMatmulScaler()) {}

    MatmulParams(Parameter<Tweight> weight_param,
                 Parameter<float> scaler_param)
        : weight(std::move(weight_param)),
          scaler(std::move(scaler_param)) {}
};

template <typename Tact, typename Tweight = Tact>
class MatmulLayer : public Layer<Tact> {
public:
    using ParamType = MatmulParams<Tweight>;
    using Layer<Tact>::forward;

    explicit MatmulLayer(eUTIL::DeviceType device,
                         ParamType params,
                         int32_t group_size = 1)
        : Layer<Tact>(device),
          m_params(std::move(params)),
          m_groupSize(group_size) {}

    explicit MatmulLayer(eUTIL::DeviceType device = eUTIL::DeviceType::kCpu)
        : Layer<Tact>(device),
          m_params(),
          m_groupSize(1) {}

    explicit MatmulLayer(eUTIL::DeviceType device,
                         Parameter<Tweight> weight,
                         Parameter<float> scaler,
                         int32_t group_size = 1)
        : Layer<Tact>(device),
          m_params(std::move(weight), std::move(scaler)),
          m_groupSize(group_size) {}

    void forward(const ForwardContext& ctx,
                 TensorListView<Tact> inputs,
                 eUTIL::Tensor<Tact>& output) override;

    void to(eUTIL::DeviceType device) override {
        this->m_device = device;
        m_params.weight.to(device);
        m_params.scaler.to(device);
    }

    void setParams(ParamType params) { m_params = std::move(params); }
    void setWeight(Parameter<Tweight> weight) { m_params.weight = std::move(weight); }
    void setScaler(Parameter<float> scaler) { m_params.scaler = std::move(scaler); }
    void setGroupSize(int32_t group_size) { m_groupSize = group_size; }
    int32_t groupSize() const { return m_groupSize; }

    const ParamType& params() const { return m_params; }
    ParamType& params() { return m_params; }

private:
    ParamType m_params;
    int32_t m_groupSize = 1;
};

}  // namespace eCEL
