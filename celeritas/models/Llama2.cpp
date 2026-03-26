#include "Llama2.h"

#include <stdexcept>

namespace eCEL {

std::vector<float> Llama2::forward(const ModelInputs& inputs) const
{
    if (m_embLayer == nullptr) {
        throw std::runtime_error("embedding layer is not initialized");
    }
    if (inputs.m_inputIds.empty()) {
        throw std::invalid_argument("Llama2::forward requires non-empty input_ids");
    }
    if (static_cast<int32_t>(inputs.m_inputIds.size()) > m_config.seq_len_) {
        throw std::invalid_argument("input_ids exceed model sequence length");
    }

    const std::size_t tokenCount = inputs.m_inputIds.size();
    const std::size_t hiddenSize = static_cast<std::size_t>(m_config.dim_);

    Tensor input(eUTIL::DeviceType::kCpu, tokenCount);
    for (std::size_t i = 0; i < tokenCount; ++i) {
        input[static_cast<int>(i)] = static_cast<float>(inputs.m_inputIds[i]);
    }

    Tensor output(m_device, tokenCount, hiddenSize);
    if (m_device == eUTIL::DeviceType::kCuda) {
        input.cuda();
    }

    ForwardContext ctx;
    m_embLayer->forward(ctx, input, output);

    if (m_device == eUTIL::DeviceType::kCuda) {
        output.cpu();
    }

    std::vector<float> embeddings(output.size());
    const float* outputData = output.data();
    for (std::size_t i = 0; i < output.size(); ++i) {
        embeddings[i] = outputData[i];
    }

    return embeddings;
}

void Llama2::createLayers()
{
    createEmb();
}

void Llama2::createEmb()
{
    if (m_rawData == nullptr) {
        throw std::runtime_error("raw model data is not initialized");
    }
    if (m_config.vocab_size_ <= 0 || m_config.dim_ <= 0 || m_config.seq_len_ <= 0) {
        throw std::runtime_error("invalid embedding config in checkpoint");
    }

    auto* weight_ptr = const_cast<float*>(static_cast<const float*>(m_rawData->weight(0)));
    eUTIL::Tensor<float> weight(eUTIL::DeviceType::kCpu,
                                {static_cast<std::size_t>(m_config.vocab_size_),
                                 static_cast<std::size_t>(m_config.dim_)},
                                weight_ptr,
                                true);
    if (m_device == eUTIL::DeviceType::kCuda) {
        weight.cuda();
    }

    m_embInput = std::make_unique<Tensor>(m_device, static_cast<std::size_t>(m_config.seq_len_));
    m_embOutput = std::make_unique<Tensor>(m_device,
                                           static_cast<std::size_t>(m_config.seq_len_),
                                           static_cast<std::size_t>(m_config.dim_));
    m_embLayer = std::make_unique<EmbLayer<float, float>>(
        m_device,
        Parameter<float>(std::move(weight), "tok_embeddings.weight"),
        m_config.vocab_size_);
}

}  // namespace eCEL
