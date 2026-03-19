#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include "celeritas/kernels/KernelFactory.h"
#include "udm/core/Tensor.h"

namespace {

constexpr int32_t kVocabSize = 3;
constexpr int32_t kEmbDim = 4;
constexpr int32_t kTokenNum = 3;

void fillEmbeddingInputs(eUTIL::Tensor<float>& input, eUTIL::Tensor<float>& weight) {
    input[0] = 2.f;
    input[1] = 0.f;
    input[2] = 1.f;

    const float weightData[kVocabSize * kEmbDim] = {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 12.f
    };
    for (int i = 0; i < kVocabSize * kEmbDim; ++i) {
        weight[i] = weightData[i];
    }
}

void expectEmbeddingOutput(const eUTIL::Tensor<float>& output) {
    const float expected[kTokenNum * kEmbDim] = {
        9.f, 10.f, 11.f, 12.f,
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f
    };
    for (int i = 0; i < kTokenNum * kEmbDim; ++i) {
        ASSERT_NEAR(output.data()[i], expected[i], 1e-5f);
    }
}

}  // namespace

TEST(test_emb, emb_cpu_matches_reference) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, kTokenNum);
    Tensor<float> weight(DeviceType::kCpu, kVocabSize, kEmbDim);
    Tensor<float> output(DeviceType::kCpu, kTokenNum, kEmbDim);
    fillEmbeddingInputs(input, weight);

    auto embKernel = eCEL::KernelFactory::getEmbKernel();
    embKernel(input, weight, output, kVocabSize, nullptr);

    expectEmbeddingOutput(output);
}

TEST(test_emb, emb_gpu_matches_cpu) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        GTEST_SKIP() << "CUDA device not available";
    }

    Tensor<float> cpuInput(DeviceType::kCpu, kTokenNum);
    Tensor<float> cpuWeight(DeviceType::kCpu, kVocabSize, kEmbDim);
    Tensor<float> cpuOutput(DeviceType::kCpu, kTokenNum, kEmbDim);
    fillEmbeddingInputs(cpuInput, cpuWeight);

    auto embKernel = eCEL::KernelFactory::getEmbKernel();
    embKernel(cpuInput, cpuWeight, cpuOutput, kVocabSize, nullptr);

    Tensor<float> gpuInput = cpuInput.cuda();
    Tensor<float> gpuWeight = cpuWeight.cuda();
    Tensor<float> gpuOutput(DeviceType::kCuda, kTokenNum, kEmbDim);
    embKernel(gpuInput, gpuWeight, gpuOutput, kVocabSize, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    gpuOutput.cpu();

    for (int i = 0; i < kTokenNum * kEmbDim; ++i) {
        ASSERT_NEAR(cpuOutput[i], gpuOutput[i], 1e-5f);
    }
    expectEmbeddingOutput(gpuOutput);
}
