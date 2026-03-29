#pragma once

// Manages per-layer KV cache storage and lightweight layer views for inference.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "celeritas/models/config.h"
#include "celeritas/operation/Layer.h"
#include "udm/common/BaseTypes.h"
#include "udm/core/Tensor.h"

namespace eCEL {

class KVCacheView {
public:
    float* k_ptr = nullptr;
    float* v_ptr = nullptr;
    std::size_t seq_len = 0;
};

class KVCacheManager {
public:
    KVCacheManager(eUTIL::DeviceType device,
                   const TransformerConfig& config);

    void allocate();
    void free();
    KVCacheView* getLayerView(int32_t layerId);
    const KVCacheView* getLayerView(int32_t layerId) const;
    void appendTokens(std::size_t tokenCount);

    bool isAllocated() const { return !m_layerViews.empty(); }
    std::size_t seqLen() const { return m_seqLen; }

private:
    // Ensures the layer index is valid before cache access.
    void validateLayerId(int32_t layerId) const;
    // Refreshes the exposed view pointers and sequence length.
    void syncLayerView(int32_t layerId);

private:
    eUTIL::DeviceType m_device;
    TransformerConfig m_config;
    std::size_t m_seqLen = 0;
    std::vector<eUTIL::Tensor<float>> m_kCaches;
    std::vector<eUTIL::Tensor<float>> m_vCaches;
    std::vector<KVCacheView> m_layerViews;
};

}  // namespace eCEL
