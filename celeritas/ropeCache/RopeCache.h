#pragma once

// Defines a lightweight RoPE cache and read-only runtime view for attention.

#include <cstddef>
#include <cstdint>

#include "celeritas/models/config.h"
#include "udm/common/BaseTypes.h"
#include "udm/core/Tensor.h"

namespace eCEL {

class RopeCacheView {
public:
    const float* cos_ptr = nullptr;
    const float* sin_ptr = nullptr;
    std::size_t seq_len = 0;
    int32_t head_size = 0;
};

class RopeCache {
public:
    RopeCache(eUTIL::DeviceType device,
              const TransformerConfig& config,
              float theta = 0.0f);

    void build();
    void clear();

    bool isBuilt() const { return m_view.cos_ptr != nullptr && m_view.sin_ptr != nullptr; }
    const RopeCacheView& view() const { return m_view; }
    float theta() const { return m_theta; }

private:
    // Validates cached shape values before building tensors.
    void validateConfig() const;
    // Refreshes the exposed cache pointers after tensor rebuild.
    void syncView();

private:
    eUTIL::DeviceType m_device;
    TransformerConfig m_config;
    float m_theta = 0.0f;
    eUTIL::Tensor<float> m_cosCache;
    eUTIL::Tensor<float> m_sinCache;
    RopeCacheView m_view;
};

}  // namespace eCEL
