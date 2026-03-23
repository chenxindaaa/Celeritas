#include <cmath>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "celeritas/kernels/KernelFactory.h"
#include "udm/core/Tensor.h"

namespace {

float swigluReference(float input1, float input2) {
    const float sigmoid = 1.0f / (1.0f + std::exp(-input1));
    return input1 * sigmoid * input2;
}

void fillSwigluInputs(eUTIL::Tensor<float>& input1, eUTIL::Tensor<float>& input2) {
    const float input1Data[] = {-3.0f, -1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 3.0f, 6.0f};
    const float input2Data[] = {2.0f, -4.0f, 1.5f, 3.0f, -2.0f, 0.5f, 1.0f, -1.0f};

    for (int i = 0; i < 8; ++i) {
        input1[i] = input1Data[i];
        input2[i] = input2Data[i];
    }
}

}  // namespace

TEST(test_swiglu, swiglu_cpu_matches_reference) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input1(DeviceType::kCpu, 8);
    Tensor<float> input2(DeviceType::kCpu, 8);
    Tensor<float> output(DeviceType::kCpu, 8);
    fillSwigluInputs(input1, input2);

    auto kernel = eCEL::KernelFactory::getSwigluKernel();
    kernel(input1, input2, output, nullptr);

    for (int i = 0; i < 8; ++i) {
        ASSERT_NEAR(output[i], swigluReference(input1[i], input2[i]), 1e-5f);
    }
}

TEST(test_swiglu, swiglu_gpu_matches_cpu) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        GTEST_SKIP() << "CUDA device not available";
    }

    Tensor<float> cpuInput1(DeviceType::kCpu, 8);
    Tensor<float> cpuInput2(DeviceType::kCpu, 8);
    Tensor<float> cpuOutput(DeviceType::kCpu, 8);
    fillSwigluInputs(cpuInput1, cpuInput2);

    auto kernel = eCEL::KernelFactory::getSwigluKernel();
    kernel(cpuInput1, cpuInput2, cpuOutput, nullptr);

    Tensor<float> gpuInput1 = cpuInput1.cuda();
    Tensor<float> gpuInput2 = cpuInput2.cuda();
    Tensor<float> gpuOutput(DeviceType::kCuda, 8);
    kernel(gpuInput1, gpuInput2, gpuOutput, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    gpuOutput.cpu();
    for (int i = 0; i < 8; ++i) {
        ASSERT_NEAR(gpuOutput[i], cpuOutput[i], 1e-5f);
    }
}
