#include <udm/core/Tensor.h>

#include "swiglu_kernel.cuh"
namespace eCEL {
__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) {
        return;
    }
    extern __shared__ float shared_mem[];
    float* smem1 = shared_mem;
    float* smem2 = shared_mem + blockDim.x;

    smem1[tid] = in1[idx];
    smem2[tid] = in2[idx];
    __syncthreads();

    float value = 1.0f / (1.0f + exp(-smem1[tid]));
    smem1[tid] = smem1[tid] * value;

    out[idx] = smem1[tid] * smem2[tid];
}

template <>
void swigluKernelCu(const eUTIL::Tensor<float>& input1, const eUTIL::Tensor<float>& input2,
                    eUTIL::Tensor<float>& output, void* stream) {
    // CHECK_EQ(input1.is_empty(), false);
    // CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

    // CHECK_EQ(input2.is_empty(), false);
    // CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

    // CHECK_EQ(output.is_empty(), false);
    // CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

    int size = static_cast<int32_t>(input1.size());
    int threads = 128;
    int blocks = (size + threads - 1) / threads;
    const size_t shmem = threads * sizeof(float) * 2;
    if (!stream) {
        swiglu_kernel_cu_fp32<<<blocks, threads, shmem>>>(
            size, input1.data(), input2.data(), output.data());
    } else {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        swiglu_kernel_cu_fp32<<<blocks, threads, shmem, stream_>>>(
            size, input1.data(), input2.data(), output.data());
    }
}
}  // namespace eCEL
