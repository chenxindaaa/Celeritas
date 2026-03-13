#include "add.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

namespace {

__global__ void AddKernel(const int* a, const int* b, int* o, int n) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        o[idx] = a[idx] + b[idx];
    }
}

void CheckCuda(cudaError_t code, const char* op) {
    if (code != cudaSuccess) {
        throw std::runtime_error(std::string(op) + " failed: " +
                                 cudaGetErrorString(code));
    }
}

}  // namespace

namespace eCEL {

void add_kernel_cu(const eUTIL::Tensor<int>& input1,
                   const eUTIL::Tensor<int>& input2,
                   eUTIL::Tensor<int>& output, void* stream) {
    const int32_t size = static_cast<int32_t>(input1.size());
    if (size != static_cast<int32_t>(input2.size()) ||
        size != static_cast<int32_t>(output.size())) {
        throw std::invalid_argument("input/output size mismatch in add_kernel_cu");
    }

    constexpr int32_t thread_num = 512;
    const int32_t block_num = (size + thread_num - 1) / thread_num;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    AddKernel<<<block_num, thread_num, 0, cuda_stream>>>(input1.data(), input2.data(),
                                                          output.data(), size);
    CheckCuda(cudaPeekAtLastError(), "AddKernel launch");
}

}  // namespace eCEL
