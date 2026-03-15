#include <gtest/gtest.h>
#include <cstdlib>

#include <cuda_runtime_api.h>

#include "baseutil/tensor/tensor.h"
#include "celeritas/kernels/KernelFactory.h"

constexpr std::size_t kCount = 25;

TEST(test_add, gpu_add) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;


    Tensor<int> cpuA(DeviceType::kCpu, kCount);
    Tensor<int> cpuB(DeviceType::kCpu, kCount);
    Tensor<int> cpuO(DeviceType::kCpu, kCount);

    // Initialize
    for (int i = 0; i < kCount; ++i) {
        cpuA[i] = std::rand() % 10;
        cpuB[i] = std::rand() % 10;
    }

    auto addKernelCpu = eCEL::KernelFactory::getAddKernel();
    addKernelCpu(cpuA, cpuB, cpuO, nullptr);

    Tensor<int> gpuA = cpuA.cuda();
    Tensor<int> gpuB = cpuB.cuda();
    Tensor<int> gpuO(DeviceType::kCuda, kCount);

    auto addKernelCuda = eCEL::KernelFactory::getAddKernel();
    addKernelCuda(gpuA, gpuB, gpuO, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    gpuO.cpu();
    for (int i = 0; i < kCount; ++i) {
        EXPECT_EQ(cpuO[i], gpuO[i]);
    }
}
