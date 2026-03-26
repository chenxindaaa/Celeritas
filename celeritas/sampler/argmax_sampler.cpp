#include "./argmax_sampler.h"
#include "celeritas/kernels/KernelFactory.h"

namespace sampler {
size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
  auto kernel = KernelFactory::getAddKernel();
  kernel(input, m_params.value.view(), output, nullptr);
}
}  // namespace sampler