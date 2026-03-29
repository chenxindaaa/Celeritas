#include "RopeCache.h"

#include <stdexcept>

#include "celeritas/kernels/KernelFactory.h"

namespace eCEL {

namespace {

float defaultRopeTheta() {
#if defined(LLAMA3_SUPPORT)
    return 500000.0f;
#elif defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    return 1000000.0f;
#else
    return 10000.0f;
#endif
}

}  // namespace

RopeCache::RopeCache(eUTIL::DeviceType device,
                     const TransformerConfig& config,
                     float theta)
    : m_device(device),
      m_config(config),
      m_theta(theta > 0.0f ? theta : defaultRopeTheta()),
      m_cosCache(device),
      m_sinCache(device) {
    validateConfig();
}

void RopeCache::build() {
    clear();

    if (m_config.seqLen == 0 || m_config.headSize == 0) {
        return;
    }

    const std::size_t seqLen = static_cast<std::size_t>(m_config.seqLen);
    const std::size_t headSize = static_cast<std::size_t>(m_config.headSize);

    eUTIL::Tensor<float> cosCache(m_device, seqLen, headSize);
    eUTIL::Tensor<float> sinCache(m_device, seqLen, headSize);
    KernelFactory::getRopeCacheKernel()(m_config.headSize,
                                        m_config.seqLen,
                                        m_theta,
                                        sinCache,
                                        cosCache);

    m_sinCache = std::move(sinCache);
    m_cosCache = std::move(cosCache);
    syncView();
}

void RopeCache::clear() {
    m_cosCache = eUTIL::Tensor<float>(m_device);
    m_sinCache = eUTIL::Tensor<float>(m_device);
    m_view = {};
    m_view.head_size = m_config.headSize;
    m_view.seq_len = 0;
}

void RopeCache::validateConfig() const {
    if (m_config.seqLen < 0 || m_config.headSize < 0) {
        throw std::invalid_argument("RopeCache config values must not be negative");
    }
    if (m_theta <= 0.0f) {
        throw std::invalid_argument("RopeCache theta must be greater than 0");
    }
}

void RopeCache::syncView() {
    m_view.cos_ptr = m_cosCache.empty() ? nullptr : m_cosCache.data();
    m_view.sin_ptr = m_sinCache.empty() ? nullptr : m_sinCache.data();
    m_view.seq_len = m_cosCache.empty() ? 0 : static_cast<std::size_t>(m_config.seqLen);
    m_view.head_size = m_config.headSize;
}

}  // namespace eCEL
