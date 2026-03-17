#include <cmath>
#include <random>
#include <gtest/gtest.h>

#include "udm/core/Tensor.h"
#include "celeritas/kernels/KernelFactory.h"

constexpr std::size_t kCount = 10000;

TEST(test_rmsnorm, rmsnorm_cpu_matches_reference) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> cpuI(DeviceType::kCpu, kCount);
    Tensor<float> cpuW(DeviceType::kCpu, kCount);
    Tensor<float> cpuO(DeviceType::kCpu, kCount);
    Tensor<float> refO(DeviceType::kCpu, kCount);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (std::size_t i = 0; i < kCount; ++i) {
        cpuI[static_cast<int>(i)] = dist(rng);
        cpuW[static_cast<int>(i)] = dist(rng);
    }

    auto rmsnormKernel = eCEL::KernelFactory::getRmsKernel();
    rmsnormKernel(cpuI, cpuW, cpuO, nullptr);

    float meanSquare = 0.f;
    for (std::size_t i = 0; i < kCount; ++i) {
        const float v = cpuI[static_cast<int>(i)];
        meanSquare += v * v;
    }
    meanSquare /= static_cast<float>(kCount);
    #if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    const float eps = 1e-6f;
    #else
    const float eps = 1e-5f;
    #endif
    const float invRms = 1.f / std::sqrt(meanSquare + eps);
    for (std::size_t i = 0; i < kCount; ++i) {
        refO[static_cast<int>(i)] =
            cpuW[static_cast<int>(i)] * (cpuI[static_cast<int>(i)] * invRms);
    }

    for (std::size_t i = 0; i < kCount; ++i) {
        ASSERT_NEAR(cpuO[static_cast<int>(i)], refO[static_cast<int>(i)], 1e-5f);
    }
}

TEST(test_rmsnorm, rmsnorm_gpu_matches_cpu) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> cpuI(DeviceType::kCpu, kCount);
    Tensor<float> cpuW(DeviceType::kCpu, kCount);
    Tensor<float> cpuO(DeviceType::kCpu, kCount);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (std::size_t i = 0; i < kCount; ++i) {
        cpuI[static_cast<int>(i)] = dist(rng);
        cpuW[static_cast<int>(i)] = dist(rng);
    }

    auto rmsnormKernel = eCEL::KernelFactory::getRmsKernel();
    rmsnormKernel(cpuI, cpuW, cpuO, nullptr);
 
    Tensor<float> gpuI = cpuI.cuda();
    Tensor<float> gpuW = cpuW.cuda();
    Tensor<float> gpuO(DeviceType::kCuda, kCount);
    rmsnormKernel(gpuI, gpuW, gpuO, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    gpuO.cpu();

    for (std::size_t i = 0; i < kCount; ++i) {
        ASSERT_NEAR(cpuO[static_cast<int>(i)], gpuO[static_cast<int>(i)], 1e-5f);
    }
}


