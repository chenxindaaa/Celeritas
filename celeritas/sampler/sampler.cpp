// Defines the common sampler base used by text generation.

#include "sampler.h"

namespace eCEL {

Sampler::Sampler(eUTIL::DeviceType device)
    : m_device(device) {}

}  // namespace eCEL
