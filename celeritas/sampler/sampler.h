#pragma once

#include <cstddef>
#include <vector>

#include "udm/common/BaseTypes.h"
#include "udm/common/cudaConfig.h"

namespace eCEL {

class Sampler {
public:
    explicit Sampler(eUTIL::DeviceType device);
    virtual ~Sampler() = default;

    virtual std::size_t sample(const std::vector<float>& logits,
                               eUTIL::CudaConfig* config = nullptr) const = 0;

    eUTIL::DeviceType device() const { return m_device; }

protected:
    eUTIL::DeviceType m_device;
};

}  // namespace eCEL
