#include <gtest/gtest.h>
#include <cstdlib>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "udm/core/Tensor.h"
#include "celeritas/kernels/KernelFactory.h"

constexpr std::size_t kCount = 25;

TEST(test_add, gpu_add) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;


    Tensor<float> cpuA(DeviceType::kCpu, kCount);
    Tensor<float> cpuB(DeviceType::kCpu, kCount);
    Tensor<float> cpuO(DeviceType::kCpu, kCount);

    // Initialize
    for (int i = 0; i < kCount; ++i) {
        cpuA[i] = static_cast<float>(std::rand() % 10);
        cpuB[i] = static_cast<float>(std::rand() % 10);
    }

    auto addKernelCpu = eCEL::KernelFactory::getAddKernel();
    addKernelCpu(cpuA, cpuB, cpuO, nullptr);

    Tensor<float> gpuA = cpuA.cuda();
    Tensor<float> gpuB = cpuB.cuda();
    Tensor<float> gpuO(DeviceType::kCuda, kCount);

    auto addKernelCuda = eCEL::KernelFactory::getAddKernel();
    addKernelCuda(gpuA, gpuB, gpuO, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    gpuO.cpu();
    for (int i = 0; i < kCount; ++i) {
        EXPECT_EQ(cpuO[i], gpuO[i]);
    }
}


