#include "DecoderLayer.h"

#include <stdexcept>

namespace eCEL {

namespace {

template <typename T>
eUTIL::Tensor<T> makeTensorLike(const eUTIL::Tensor<T>& input,
                                eUTIL::DeviceType device) {
    if (input.dimSize() == 1) {
        return eUTIL::Tensor<T>(device, static_cast<std::size_t>(input.getDim(0)));
    }
    if (input.dimSize() == 2) {
        return eUTIL::Tensor<T>(device,
                                static_cast<std::size_t>(input.getDim(0)),
                                static_cast<std::size_t>(input.getDim(1)));
    }
    throw std::invalid_argument("DecoderLayer only supports 1D or 2D tensors");
}

}  // namespace

template <typename Tact, typename Tweight>
void DecoderLayer<Tact, Tweight>::forward(const ForwardContext& ctx,
                                          TensorListView<Tact> inputs,
                                          eUTIL::Tensor<Tact>& output) {
    Layer<Tact>::requireInputCount(inputs, 1, "DecoderLayer");

    const eUTIL::Tensor<Tact>& input = inputs[0];
    eUTIL::Tensor<Tact> attnNormOutput = makeTensorLike(input, this->m_device);
    m_attnNormLayer.forward(ctx, input, attnNormOutput);

    eUTIL::Tensor<Tact> attnOutput = makeTensorLike(input, this->m_device);
    m_selfAttentionLayer.forward(ctx, attnNormOutput, attnOutput);

    eUTIL::Tensor<Tact> hiddenState = makeTensorLike(input, this->m_device);
    const eUTIL::Tensor<Tact>* attnAddInputs[] = {&input, &attnOutput};
    m_addLayer.forward(ctx, TensorListView<Tact>(attnAddInputs, 2), hiddenState);

    eUTIL::Tensor<Tact> ffnNormOutput = makeTensorLike(hiddenState, this->m_device);
    m_ffnNormLayer.forward(ctx, hiddenState, ffnNormOutput);

    eUTIL::Tensor<Tact> mlpOutput = makeTensorLike(hiddenState, this->m_device);
    m_mlpLayer.forward(ctx, ffnNormOutput, mlpOutput);

    const eUTIL::Tensor<Tact>* ffnAddInputs[] = {&hiddenState, &mlpOutput};
    m_addLayer.forward(ctx, TensorListView<Tact>(ffnAddInputs, 2), output);
}

template class DecoderLayer<float, float>;

}  // namespace eCEL
