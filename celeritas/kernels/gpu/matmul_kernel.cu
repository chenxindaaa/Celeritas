#include "matmul_kernel.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

namespace {

__global__ void matmulKernelFp32(const float* input,
                                 const float* weight,
                                 float* output,
                                 int32_t inputRows,
                                 int32_t sharedDim,
                                 int32_t outputCols,
                                 float scale) {
    const int32_t col = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t row = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row >= inputRows || col >= outputCols) {
        return;
    }

    float sum = 0.0f;
    for (int32_t k = 0; k < sharedDim; ++k) {
        sum += input[row + k * inputRows] * weight[k + col * sharedDim];
    }
    output[row + col * inputRows] = sum * scale;
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
void matmulKernelCu(const eUTIL::Tensor<float>& input,
                    const eUTIL::Tensor<float>& weight,
                    eUTIL::Tensor<float>& output,
                    const float scale,
                    const eUTIL::CudaConfig* config) {
    if (input.empty() || weight.empty() || output.empty()) {
        throw std::invalid_argument("matmulKernelCu received empty tensor");
    }
    if (input.device() != eUTIL::DeviceType::kCuda ||
        weight.device() != eUTIL::DeviceType::kCuda ||
        output.device() != eUTIL::DeviceType::kCuda) {
        throw std::invalid_argument("matmulKernelCu requires CUDA tensors");
    }
    if (weight.dimSize() != 2) {
        throw std::invalid_argument("matmulKernelCu requires weight to be 2D");
    }

    int32_t inputCols = 1;
    int32_t inputRows = 1;
    if (input.dimSize() == 2) {
        inputCols = input.getDim(0);
        inputRows = input.getDim(1);
    } else if (input.dimSize() == 1) {
        inputCols = input.getDim(0);
    } else {
        throw std::invalid_argument("matmulKernelCu requires input to be 1D or 2D");
    }

    const int32_t outputCols = weight.getDim(0);
    const int32_t sharedDim = weight.getDim(1);
    if (inputCols != sharedDim) {
        throw std::invalid_argument("matmulKernelCu input/weight shape mismatch");
    }
    if (output.size() != static_cast<std::size_t>(inputRows) * static_cast<std::size_t>(outputCols)) {
        throw std::invalid_argument("matmulKernelCu output shape mismatch");
    }

    constexpr int32_t blockX = 16;
    constexpr int32_t blockY = 16;
    const dim3 block(blockX, blockY);
    const dim3 grid((outputCols + blockX - 1) / blockX,
                    (inputRows + blockY - 1) / blockY);

    cudaStream_t stream = config ? config->stream : nullptr;
    if (stream != nullptr) {
        matmulKernelFp32<<<grid, block, 0, stream>>>(input.data(), weight.data(), output.data(),
                                                     inputRows, sharedDim, outputCols, scale);
    } else {
        matmulKernelFp32<<<grid, block>>>(input.data(), weight.data(), output.data(),
                                          inputRows, sharedDim, outputCols, scale);
    }
    checkCuda(cudaPeekAtLastError(), "matmulKernelFp32 launch");
}

}  // namespace eCEL
