#include "MlpLayer.h"

#include <stdexcept>

namespace eCEL {

template <typename Tact, typename Tweight>
Tensor MlpLayer<Tact, Tweight>::makeMatmulOutputTensor(const Tensor& input,
                                                       const eUTIL::Tensor<Tweight>& weight,
                                                       eUTIL::DeviceType device) {
    if (weight.dimSize() != 2) {
        throw std::invalid_argument("MlpLayer matmul weight must be 2D");
    }

    const std::size_t out_dim = static_cast<std::size_t>(weight.getDim(0));
    if (input.dimSize() == 1) {
        return Tensor(device, out_dim);
    }
    if (input.dimSize() == 2) {
        return Tensor(device, out_dim, static_cast<std::size_t>(input.getDim(1)));
    }
    throw std::invalid_argument("MlpLayer only supports 1D or 2D input tensors");
}

template <typename Tact, typename Tweight>
void MlpLayer<Tact, Tweight>::forward(const ForwardContext& ctx,
                                      const Tensor& input,
                                      Tensor& output) {
    Tensor gate_output = makeMatmulOutputTensor(input, m_params.gate_proj.weight.view(), m_device);
    Tensor up_output = makeMatmulOutputTensor(input, m_params.up_proj.weight.view(), m_device);

    m_gateLayer.forward(ctx, input, gate_output);
    m_upLayer.forward(ctx, input, up_output);

    Tensor swiglu_output = makeMatmulOutputTensor(input, m_params.gate_proj.weight.view(), m_device);
    m_swigluLayer.params().value = Parameter<float>(up_output, "up_proj_output");
    m_swigluLayer.forward(ctx, gate_output, swiglu_output);

    Tensor down_output = makeMatmulOutputTensor(swiglu_output, m_params.down_proj.weight.view(), m_device);
    m_downLayer.forward(ctx, swiglu_output, down_output);

    m_addLayer.params().value = Parameter<float>(input, "residual");
    m_addLayer.forward(ctx, down_output, output);
}

template class MlpLayer<float, float>;

}  // namespace eCEL
