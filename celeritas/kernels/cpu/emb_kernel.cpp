#include "emb_kernel.h"

#include <cstring>
#include <stdexcept>

namespace eCEL {

void embKernelCpu(const eUTIL::Tensor<int>& input,
                  const eUTIL::Tensor<float>& weight,
                  eUTIL::Tensor<float>& output,
                  int32_t vocabSize,
                  void* stream) {
    (void)stream;
    if (input.empty() || weight.empty() || output.empty()) {
        throw std::invalid_argument("embKernelCpu received empty tensor");
    }

    if (input.device() != eUTIL::DeviceType::kCpu ||
        weight.device() != eUTIL::DeviceType::kCpu ||
        output.device() != eUTIL::DeviceType::kCpu) {
        throw std::invalid_argument("embKernelCpu requires CPU tensors");
    }

    if (weight.dimSize() == 0) {
        throw std::invalid_argument("Embedding weight dim must be greater than 0");
    }

    const std::size_t tokenNum = input.size();
    const std::size_t embDim = weight.dims().back();
    if (weight.size() < static_cast<std::size_t>(vocabSize) * embDim) {
        throw std::invalid_argument("Embedding weight size is smaller than vocab_size * dim");
    }
    if (output.size() != tokenNum * embDim) {
        throw std::invalid_argument("Embedding output size mismatch");
    }

    const int* inputPtr = input.data();
    const float* weightPtr = weight.data();
    float* outputPtr = output.data();

    for (std::size_t i = 0; i < tokenNum; ++i) {
        const int token = inputPtr[i];
        if (token < 0 || token >= vocabSize) {
            throw std::out_of_range("Token index out of range");
        }
        const float* src = weightPtr + static_cast<std::size_t>(token) * embDim;
        float* dst = outputPtr + i * embDim;
        std::memcpy(dst, src, embDim * sizeof(float));
    }
}

}  // namespace eCEL
