#include "sampler.h"
#include "sampler.h"

class CudaConfig;

namespace eCEL {

class ArgmaxSampler : public Sampler {
 public:
  explicit ArgmaxSampler(eUTIL::DeviceType device) : Sampler(device) {}

  size_t sample(const float* logits, size_t size, CudaConfig* config) override;
};

}  // namespace eCEL