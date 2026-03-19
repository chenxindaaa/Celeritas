#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include "celeritas/kernels/KernelRegistry.h"
#include "udm/core/Tensor.h"

namespace {

void fillMatmulInputs(eUTIL::Tensor<float>& input, eUTIL::Tensor<float>& weight) {
    const float inputData[] = {1.f, 4.f, 2.f, 5.f, 3.f, 6.f};
    const float weightData[] = {7.f, 9.f, 11.f, 8.f, 10.f, 12.f};

    for (int i = 0; i < 6; ++i) {
        input[i] = inputData[i];
        weight[i] = weightData[i];
    }
}

void expectMatmulResult(const eUTIL::Tensor<float>& output, float scale = 1.f) {
    const float expected[] = {58.f * scale, 139.f * scale, 64.f * scale, 154.f * scale};
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(output.data()[i], expected[i], 1e-5f);
    }
}

}  // namespace

TEST(test_matmul, matmul_cpu_matches_reference) {
    using eUTIL::DType;
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, 3, 2);
    Tensor<float> weight(DeviceType::kCpu, 2, 3);
    Tensor<float> output(DeviceType::kCpu, 2, 2);
    fillMatmulInputs(input, weight);

    auto kernel = eCEL::KernelRegistry::getInstance().lookup<eCEL::MatmulKernelFn<float>>(
        eCEL::OpType::kMatmul, DeviceType::kCpu, DType::kFloat32);
    kernel(input, weight, output, 1.f, nullptr);

    expectMatmulResult(output);
}

TEST(test_matmul, matmul_gpu_matches_cpu) {
    using eUTIL::DType;
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        GTEST_SKIP() << "CUDA device not available";
    }

    Tensor<float> cpuInput(DeviceType::kCpu, 3, 2);
    Tensor<float> cpuWeight(DeviceType::kCpu, 2, 3);
    Tensor<float> cpuOutput(DeviceType::kCpu, 2, 2);
    fillMatmulInputs(cpuInput, cpuWeight);

    auto cpuKernel = eCEL::KernelRegistry::getInstance().lookup<eCEL::MatmulKernelFn<float>>(
        eCEL::OpType::kMatmul, DeviceType::kCpu, DType::kFloat32);
    cpuKernel(cpuInput, cpuWeight, cpuOutput, 0.5f, nullptr);

    Tensor<float> gpuInput = cpuInput.cuda();
    Tensor<float> gpuWeight = cpuWeight.cuda();
    Tensor<float> gpuOutput(DeviceType::kCuda, 2, 2);

    auto gpuKernel = eCEL::KernelRegistry::getInstance().lookup<eCEL::MatmulKernelFn<float>>(
        eCEL::OpType::kMatmul, DeviceType::kCuda, DType::kFloat32);
    gpuKernel(gpuInput, gpuWeight, gpuOutput, 0.5f, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    gpuOutput.cpu();

    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(cpuOutput[i], gpuOutput[i], 1e-5f);
    }
    expectMatmulResult(gpuOutput, 0.5f);
}
