// Defines argmax sampling for CPU and CUDA generation.

#include "argmax_sampler.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

#include "celeritas/kernels/KernelFactory.h"
#include "udm/core/Tensor.h"

namespace eCEL {

std::size_t ArgmaxSampler::sample(const std::vector<float>& logits,
                                  eUTIL::CudaConfig* config) const {
    if (logits.empty()) {
        throw std::invalid_argument("ArgmaxSampler requires non-empty logits");
    }

    if (m_device == eUTIL::DeviceType::kCpu || config == nullptr) {
        const auto bestIt = std::max_element(logits.begin(), logits.end());
        return static_cast<std::size_t>(std::distance(logits.begin(), bestIt));
    }

    if (m_device == eUTIL::DeviceType::kCuda) {
        eUTIL::Tensor<float> logitsTensor(eUTIL::DeviceType::kCpu, logits.size());
        for (std::size_t i = 0; i < logits.size(); ++i) {
            logitsTensor[static_cast<int>(i)] = logits[i];
        }
        logitsTensor.cuda(config ? config->stream : nullptr);
        eUTIL::Tensor<std::size_t> outputTensor(eUTIL::DeviceType::kCuda, 1);

        auto kernel = KernelFactory::getArgmaxKernel();
        kernel(logitsTensor, outputTensor, config ? config->stream : nullptr);
        outputTensor.cpu();
        return outputTensor.data()[0];
    }

    throw std::invalid_argument("ArgmaxSampler received unsupported device");
}

}  // namespace eCEL
