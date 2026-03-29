#include "argmax_kernel.cuh"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cuda_runtime_api.h>

namespace eCEL {
namespace {

constexpr std::size_t kInvalidArgmaxIndex = static_cast<std::size_t>(-1);

__forceinline__ __device__ void warpReduceArgmax(float& value, std::size_t& index) {
    const unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
    for (unsigned int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
        const float otherValue = __shfl_down_sync(mask, value, offset, warpSize);
        const std::size_t otherIndex = __shfl_down_sync(mask, index, offset, warpSize);
        if (index == kInvalidArgmaxIndex ||
            otherIndex == kInvalidArgmaxIndex) {
            continue;
        }
        if (otherValue > value || (otherValue == value && otherIndex < index)) {
            value = otherValue;
            index = otherIndex;
        }
    }
}

__forceinline__ __device__ void blockReduceArgmax(float& value,
                                                  std::size_t& index,
                                                  float* sharedValues,
                                                  std::size_t* sharedIndices) {
    const int laneId = threadIdx.x % warpSize;
    const int warpId = threadIdx.x / warpSize;

    warpReduceArgmax(value, index);

    __syncthreads();
    if (laneId == 0) {
        sharedValues[warpId] = value;
        sharedIndices[warpId] = index;
    }

    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize) {
        value = sharedValues[laneId];
        index = sharedIndices[laneId];
    } else {
        value = 0.0f;
        index = kInvalidArgmaxIndex;
    }

    if (warpId == 0) {
        warpReduceArgmax(value, index);
    }
}

__global__ void argmaxKernelFp32(const float* inputPtr,
                                 std::size_t size,
                                 std::size_t* outputIndex) {
    __shared__ std::size_t sharedMaxIndices[32];
    __shared__ float sharedMaxValues[32];

    const std::uint32_t tid = threadIdx.x;
    if (tid >= size) {
        return;
    }

    std::size_t maxIndex = tid;
    float maxValue = inputPtr[maxIndex];
    for (std::size_t i = tid; i < size; i += blockDim.x) {
        if (inputPtr[i] > maxValue) {
            maxIndex = i;
            maxValue = inputPtr[i];
        }
    }

    blockReduceArgmax(maxValue, maxIndex, sharedMaxValues, sharedMaxIndices);
    __syncthreads();
    if (threadIdx.x == 0) {
        *outputIndex = maxIndex;
    }
}

}  // namespace

void argmaxKernelCu(const eUTIL::Tensor<float>& input,
                    eUTIL::Tensor<std::size_t>& output,
                    void* stream) {
    if (input.device() != eUTIL::DeviceType::kCuda) {
        throw std::invalid_argument("argmaxKernelCu requires CUDA input");
    }
    if (input.empty()) {
        throw std::invalid_argument("argmaxKernelCu requires non-empty input");
    }
    if (output.device() != eUTIL::DeviceType::kCuda) {
        throw std::invalid_argument("argmaxKernelCu requires CUDA output");
    }
    if (output.size() != 1) {
        throw std::invalid_argument("argmaxKernelCu requires output tensor size 1");
    }
    if (stream == nullptr) {
        argmaxKernelFp32<<<1, 512>>>(input.data(), input.size(), output.data());
    } else {
        cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
        argmaxKernelFp32<<<1, 512, 0, cudaStream>>>(input.data(), input.size(), output.data());
    }
}

}  // namespace eCEL
