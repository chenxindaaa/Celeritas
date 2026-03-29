#pragma once

#include "sampler.h"

namespace eCEL {

class ArgmaxSampler : public Sampler {
public:
    explicit ArgmaxSampler(eUTIL::DeviceType device)
        : Sampler(device) {}

    std::size_t sample(const std::vector<float>& logits,
                       eUTIL::CudaConfig* config = nullptr) const override;
};

}  // namespace eCEL
