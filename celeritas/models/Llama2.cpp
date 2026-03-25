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
