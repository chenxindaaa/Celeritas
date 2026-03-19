#include "emb_kernel.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

namespace {

__global__ void embKernelFp32(int32_t vocabSize,
                              int32_t tokenNum,
                              int32_t embDim,
                              const float* input,
                              const float* weight,
                              float* output) {
    const int32_t tokenIdx = static_cast<int32_t>(blockIdx.x);
    if (tokenIdx >= tokenNum) {
        return;
    }

    const int32_t token = static_cast<int32_t>(input[tokenIdx]);
    if (token < 0 || token >= vocabSize) {
        return;
    }

    const float* src = weight + static_cast<std::size_t>(token) * embDim;
    float* dst = output + static_cast<std::size_t>(tokenIdx) * embDim;
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < embDim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void checkCuda(cudaError_t code, const char* op) {
    if (code != cudaSuccess) {
        throw std::runtime_error(std::string(op) + " failed: " +
                                 cudaGetErrorString(code));
    }
}

}  // namespace

namespace eCEL {

template<>
void embKernelCu(const eUTIL::Tensor<float>& input,
                 const eUTIL::Tensor<float>& weight,
                 eUTIL::Tensor<float>& output,
                 int32_t vocabSize,
                 void* stream) {
    if (input.empty() || weight.empty() || output.empty()) {
        throw std::invalid_argument("embKernelCu received empty tensor");
    }
    if (input.device() != eUTIL::DeviceType::kCuda ||
        weight.device() != eUTIL::DeviceType::kCuda ||
        output.device() != eUTIL::DeviceType::kCuda) {
        throw std::invalid_argument("embKernelCu requires CUDA tensors");
    }
    if (weight.dimSize() == 0) {
        throw std::invalid_argument("Embedding weight dim must be greater than 0");
    }

    const int32_t tokenNum = static_cast<int32_t>(input.size());
    const int32_t embDim = static_cast<int32_t>(weight.dims().back());
    if (weight.size() < static_cast<std::size_t>(vocabSize) * static_cast<std::size_t>(embDim)) {
        throw std::invalid_argument("Embedding weight size is smaller than vocab_size * dim");
    }
    if (output.size() != static_cast<std::size_t>(tokenNum) * static_cast<std::size_t>(embDim)) {
        throw std::invalid_argument("Embedding output size mismatch");
    }

    constexpr int32_t blockSize = 128;
    cudaStream_t cudaStream = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    if (cudaStream != nullptr) {
        embKernelFp32<<<tokenNum, blockSize, 0, cudaStream>>>(
            vocabSize, tokenNum, embDim, input.data(), weight.data(), output.data());
    } else {
        embKernelFp32<<<tokenNum, blockSize>>>(
            vocabSize, tokenNum, embDim, input.data(), weight.data(), output.data());
    }
    checkCuda(cudaPeekAtLastError(), "embKernelFp32 launch");
}

}  // namespace eCEL
