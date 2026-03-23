#include <cmath>

#include <gtest/gtest.h>

#include "celeritas/kernels/KernelFactory.h"
#include "udm/core/Tensor.h"

namespace {

void fillSoftmaxInput(eUTIL::Tensor<float>& input) {
    const float data[] = {1.0f, 2.0f, 3.0f, -1.0f};
    for (int i = 0; i < 4; ++i) {
        input[i] = data[i];
    }
}

void computeSoftmaxReference(const eUTIL::Tensor<float>& input,
                             eUTIL::Tensor<float>& reference) {
    const float* inputData = input.data();
    float maxValue = inputData[0];
    for (int i = 1; i < 4; ++i) {
        maxValue = std::max(maxValue, inputData[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        reference[i] = std::exp(inputData[i] - maxValue);
        sum += reference[i];
    }

    for (int i = 0; i < 4; ++i) {
        reference[i] /= sum;
    }
}

}  // namespace

TEST(test_softmax, softmax_cpu_matches_reference) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, 4);
    Tensor<float> expected(DeviceType::kCpu, 4);
    fillSoftmaxInput(input);
    computeSoftmaxReference(input, expected);

    auto kernel = eCEL::KernelFactory::getSoftmaxKernel();
    kernel(input, nullptr);

    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(input[i], expected[i], 1e-5f);
    }
}

TEST(test_softmax, softmax_cpu_outputs_normalized_probabilities) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, 5);
    const float data[] = {-10.0f, 0.0f, 10.0f, 1.0f, -3.0f};
    for (int i = 0; i < 5; ++i) {
        input[i] = data[i];
    }

    auto kernel = eCEL::KernelFactory::getSoftmaxKernel();
    kernel(input, nullptr);

    float sum = 0.0f;
    for (int i = 0; i < 5; ++i) {
        EXPECT_GE(input[i], 0.0f);
        EXPECT_LE(input[i], 1.0f);
        sum += input[i];
    }
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
}
