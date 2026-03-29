#include "rope_cache_kernel.cuh"

namespace eCEL {

__global__ void rope_cache_calc_kernel(int32_t headSize,
                                       int32_t maxSeqLen,
                                       float theta,
                                       float* sinCache,
                                       float* cosCache) {
    const int32_t idx = static_cast<int32_t>(threadIdx.x + blockIdx.x * blockDim.x);
    const int32_t total = headSize * maxSeqLen;
    if (idx >= total) {
        return;
    }

    const int32_t pos = idx / headSize;
    const int32_t headDim = idx % headSize;
    const float freq = 1.0f / powf(theta, static_cast<float>(headDim) / static_cast<float>(headSize));
    const float angle = static_cast<float>(pos) * freq;
    sinCache[idx] = sinf(angle);
    cosCache[idx] = cosf(angle);
}

template <>
void ropeCacheKernelCu<float>(int32_t headSize,
                              int32_t maxSeqLen,
                              float theta,
                              eUTIL::Tensor<float>& sinCache,
                              eUTIL::Tensor<float>& cosCache,
                              void* stream) {
    const int32_t total = headSize * maxSeqLen;
    const int32_t threads = 256;
    const int32_t blocks = (total + threads - 1) / threads;

    if (stream != nullptr) {
        cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
        rope_cache_calc_kernel<<<blocks, threads, 0, cudaStream>>>(
            headSize, maxSeqLen, theta, sinCache.data(), cosCache.data());
        return;
    }

    rope_cache_calc_kernel<<<blocks, threads>>>(
        headSize, maxSeqLen, theta, sinCache.data(), cosCache.data());
}

}  // namespace eCEL
