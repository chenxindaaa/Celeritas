#include "MlpLayer.h"

#include <stdexcept>

namespace eCEL {

template <typename Tact, typename Tweight>
eUTIL::Tensor<Tact> MlpLayer<Tact, Tweight>::makeMatmulOutputTensor(const eUTIL::Tensor<Tact>& input,
                                                                    const eUTIL::Tensor<Tweight>& weight,
                                                                    eUTIL::DeviceType device) {
    if (weight.dimSize() != 2) {
        throw std::invalid_argument("MlpLayer matmul weight must be 2D");
    }

    const std::size_t outDim = static_cast<std::size_t>(weight.getDim(0));
    if (input.dimSize() == 1) {
        return eUTIL::Tensor<Tact>(device, outDim);
    }
    if (input.dimSize() == 2) {
        return eUTIL::Tensor<Tact>(device, outDim, static_cast<std::size_t>(input.getDim(1)));
    }
    throw std::invalid_argument("MlpLayer only supports 1D or 2D input tensors");
}

template <typename Tact, typename Tweight>
void MlpLayer<Tact, Tweight>::forward(const ForwardContext& ctx,
                                      TensorListView<Tact> inputs,
                                      eUTIL::Tensor<Tact>& output) {
    Layer<Tact>::requireInputCount(inputs, 1, "MlpLayer");

    const eUTIL::Tensor<Tact>& input = inputs[0];
    eUTIL::Tensor<Tact> gateOutput = makeMatmulOutputTensor(input, m_params.gate_proj.weight.view(), this->m_device);
    eUTIL::Tensor<Tact> upOutput = makeMatmulOutputTensor(input, m_params.up_proj.weight.view(), this->m_device);

    m_gateLayer.forward(ctx, input, gateOutput);
    m_upLayer.forward(ctx, input, upOutput);

    eUTIL::Tensor<Tact> swigluOutput = makeMatmulOutputTensor(input, m_params.gate_proj.weight.view(), this->m_device);
    const eUTIL::Tensor<Tact>* swigluInputs[] = {&gateOutput, &upOutput};
    m_swigluLayer.forward(ctx, TensorListView<Tact>(swigluInputs, 2), swigluOutput);

    m_downLayer.forward(ctx, swigluOutput, output);
}

template class MlpLayer<float, float>;

}  // namespace eCEL
