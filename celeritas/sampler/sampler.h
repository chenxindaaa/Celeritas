#pragma once

#include <cstddef>
#include <cstdint>

#include "udm/common/BaseTypes.h"

namespace eCEL {

class Sampler {
 public:
  explicit Sampler(eUTIL::DeviceType device) : m_device(device) {}

  virtual size_t sample(const float* logits, size_t size, CudaConfig* config = nullptr) = 0;

 protected:
  eUTIL::DeviceType m_device;
};
}  // namespace eCEL
