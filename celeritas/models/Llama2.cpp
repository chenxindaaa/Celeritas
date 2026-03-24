#include "Llama2.h"

#include <stdexcept>

namespace eCEL {

void Llama2::createLayers()
{
    createEmb();
}

void Llama2::createEmb()
{
    if (m_rawData == nullptr) {
        throw std::runtime_error("raw model data is not loaded");
    }
    if (m_config.vocab_size <= 0 || m_config.dim <= 0) {
        throw std::runtime_error("invalid model config for embedding layer");
    }

    auto& input = createTensor<float>(
        m_device, static_cast<std::size_t>(1));
    auto& output = createTensor<float>(
        m_device, static_cast<std::size_t>(1), static_cast<std::size_t>(m_config.dim));
    auto& weight = createTensor<float>(
        eUTIL::DeviceType::kCpu,
        std::initializer_list<std::size_t>{static_cast<std::size_t>(m_config.vocab_size),
                                           static_cast<std::size_t>(m_config.dim)},
        const_cast<float*>(static_cast<const float*>(m_rawData->weight(0))),
        true);

    if (m_device == eUTIL::DeviceType::kCuda) {
        weight.cuda();
    }

    auto& layer = createLayer<eCEL::EmbeddingLayer>(m_device, m_config.vocab_size);
    layer.setInput(0, input);
    layer.setWeight(0, weight);
    layer.setOutput(0, output);
}

}  // namespace eCEL
