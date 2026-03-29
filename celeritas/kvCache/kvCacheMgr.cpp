#include "kvCacheMgr.h"

#include <stdexcept>

namespace eCEL {

KVCacheManager::KVCacheManager(eUTIL::DeviceType device,
                               const TransformerConfig& config)
    : m_device(device),
      m_config(config) {
    if (m_config.layerNum < 0 || m_config.seqLen < 0 || m_config.kvDim < 0) {
        throw std::invalid_argument("KVCacheManager config values must not be negative");
    }
}

void KVCacheManager::allocate() {
    free();

    if (m_config.layerNum == 0 || m_config.seqLen == 0 || m_config.kvDim == 0) {
        return;
    }

    const std::size_t layerCount = static_cast<std::size_t>(m_config.layerNum);
    const std::size_t maxSeqLen = static_cast<std::size_t>(m_config.seqLen);
    const std::size_t kvDim = static_cast<std::size_t>(m_config.kvDim);

    m_kCaches.reserve(layerCount);
    m_vCaches.reserve(layerCount);
    m_layerViews.resize(layerCount);

    for (std::size_t i = 0; i < layerCount; ++i) {
        m_kCaches.emplace_back(m_device, maxSeqLen, kvDim);
        m_vCaches.emplace_back(m_device, maxSeqLen, kvDim);
        syncLayerView(static_cast<int32_t>(i));
    }
}

void KVCacheManager::free() {
    m_seqLen = 0;
    m_kCaches.clear();
    m_vCaches.clear();
    m_layerViews.clear();
}

KVCacheView* KVCacheManager::getLayerView(int32_t layerId) {
    validateLayerId(layerId);
    return &m_layerViews[static_cast<std::size_t>(layerId)];
}

const KVCacheView* KVCacheManager::getLayerView(int32_t layerId) const {
    validateLayerId(layerId);
    return &m_layerViews[static_cast<std::size_t>(layerId)];
}

void KVCacheManager::appendTokens(std::size_t tokenCount) {
    if (!isAllocated()) {
        throw std::runtime_error("KV cache is not allocated");
    }

    const std::size_t maxSeqLen = static_cast<std::size_t>(m_config.seqLen);
    if (m_seqLen + tokenCount > maxSeqLen) {
        throw std::out_of_range("KV cache sequence length exceeds configured limit");
    }

    m_seqLen += tokenCount;
    for (std::size_t i = 0; i < m_layerViews.size(); ++i) {
        syncLayerView(static_cast<int32_t>(i));
    }
}

void KVCacheManager::validateLayerId(int32_t layerId) const {
    if (layerId < 0 || layerId >= m_config.layerNum) {
        throw std::out_of_range("KV cache layer index is out of range");
    }
    if (!isAllocated()) {
        throw std::runtime_error("KV cache is not allocated");
    }
}

void KVCacheManager::syncLayerView(int32_t layerId) {
    KVCacheView& view = m_layerViews[static_cast<std::size_t>(layerId)];
    view.k_ptr = m_kCaches[static_cast<std::size_t>(layerId)].data();
    view.v_ptr = m_vCaches[static_cast<std::size_t>(layerId)].data();
    view.seq_len = m_seqLen;
}

}  // namespace eCEL
