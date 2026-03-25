#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include "Layer.h"
#include "celeritas/kernels/KernelFactory.h"

namespace eCEL {

template <typename Tweight>
struct EmbParams {
    Parameter<Tweight> weight;
    int32_t vocab_size = 0;

    EmbParams(Parameter<Tweight> weight_param, int32_t vocab_size_param)
        : weight(std::move(weight_param)),
          vocab_size(vocab_size_param) {}
};

template <typename Tact, typename Tweight = Tact>
class EmbLayer : public Layer {
public:
    static_assert(std::is_same_v<Tact, float>,
                  "EmbLayer currently supports float activation tensors only");
    static_assert(std::is_same_v<Tweight, float>,
                  "EmbLayer currently supports float weight tensors only");

    using ParamType = EmbParams<Tweight>;

    explicit EmbLayer(eUTIL::DeviceType device,
                      ParamType params)
        : Layer(device),
          m_params(std::move(params)) {}

    explicit EmbLayer(eUTIL::DeviceType device,
                      Parameter<Tweight> weight,
                      int32_t vocab_size)
        : Layer(device),
          m_params(std::move(weight), vocab_size) {}

    void forward(const ForwardContext& ctx,
                 const Tensor& input,
                 Tensor& output) override;

    const ParamType& params() const { return m_params; }
    ParamType& params() { return m_params; }

private:
    ParamType m_params;
};

}  // namespace eCEL
