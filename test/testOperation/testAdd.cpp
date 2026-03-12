#include <gtest/gtest.h>
#include <cstdlib>

#include <cuda_runtime_api.h>

#include "baseutil/tensor/tensor.h"
#include "celeritas/operation/kernels/gpu/add.cuh"

TEST(test_add, gpu_add) {
    using baseutil::tensor::DeviceType;
    using baseutil::tensor::Tensor;

    constexpr std::size_t kCount = 25;

    Tensor<int> cpu_a(kCount, DeviceType::kCpu);
    Tensor<int> cpu_b(kCount, DeviceType::kCpu);
    Tensor<int> cpu_o(kCount, DeviceType::kCpu);

    int* a = cpu_a.data();
    int* b = cpu_b.data();
    int* o = cpu_o.data();
    for (int i = 0; i < static_cast<int>(kCount); ++i) {
        a[i] = std::rand() % 10;
        b[i] = std::rand() % 10;
        o[i] = a[i] + b[i];
    }

    cpu_a.cuda();
    cpu_b.cuda();
    Tensor<int> gpu_o(kCount, DeviceType::kCuda);

    kernel::add_kernel_cu(cpu_a, cpu_b, gpu_o);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    gpu_o.cpu();
    const int* out = gpu_o.data();
    for (int i = 0; i < static_cast<int>(kCount); ++i) {
        EXPECT_EQ(o[i], out[i]);
    }
}
