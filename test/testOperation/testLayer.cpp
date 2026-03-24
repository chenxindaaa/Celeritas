#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <random>

#include "celeritas/operation/Layer.h"
#include "udm/core/Tensor.h"

namespace {

constexpr int32_t kVocabSize = 3;
constexpr int32_t kEmbDim = 4;
constexpr int32_t kTokenNum = 3;
constexpr std::size_t kRmsCount = 256;

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

void fillMatmulExample(eUTIL::Tensor<float>& input, eUTIL::Tensor<float>& weight) {
    const float inputData[] = {1.f, 4.f, 2.f, 5.f, 3.f, 6.f};
    const float weightData[] = {7.f, 9.f, 11.f, 8.f, 10.f, 12.f};
    for (int i = 0; i < 6; ++i) {
        input[i] = inputData[i];
        weight[i] = weightData[i];
    }
}

}  // namespace

TEST(test_layer, add_layer_forward_cpu) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input1(DeviceType::kCpu, 4);
    Tensor<float> input2(DeviceType::kCpu, 4);
    Tensor<float> output(DeviceType::kCpu, 4);

    input1[0] = 1;
    input1[1] = 2;
    input1[2] = 3;
    input1[3] = 4;
    input2[0] = 10;
    input2[1] = 20;
    input2[2] = 30;
    input2[3] = 40;

    eCEL::AddLayer layer(DeviceType::kCpu);
    EXPECT_EQ(layer.type(), eCEL::LayerType::kAdd);
    EXPECT_FALSE(layer.hasWeight());
    EXPECT_EQ(layer.inputCount(), 2U);
    EXPECT_EQ(layer.outputCount(), 1U);
    EXPECT_EQ(layer.weightCount(), 0U);

    layer.setInput(0, input1);
    layer.setInput(1, input2);
    layer.setOutput(0, output);

    EXPECT_EQ(layer.forward(), eCEL::Status::kOk);
    EXPECT_EQ(output[0], 11);
    EXPECT_EQ(output[1], 22);
    EXPECT_EQ(output[2], 33);
    EXPECT_EQ(output[3], 44);
}

TEST(test_layer, add_layer_missing_input_returns_invalid_argument) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<int8_t> input1(DeviceType::kCpu, 2);
    Tensor<int8_t> output(DeviceType::kCpu, 2);

    eCEL::AddLayer layer(DeviceType::kCpu);
    layer.setInput(0, input1);
    layer.setOutput(0, output);

    EXPECT_EQ(layer.forward(), eCEL::Status::kInvalidArgument);
}

TEST(test_layer, embedding_layer_forward_cpu) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, kTokenNum);
    Tensor<float> weight(DeviceType::kCpu, kVocabSize, kEmbDim);
    Tensor<float> output(DeviceType::kCpu, kTokenNum, kEmbDim);
    fillEmbeddingInputs(input, weight);

    eCEL::EmbeddingLayer layer(DeviceType::kCpu, kVocabSize);
    EXPECT_EQ(layer.type(), eCEL::LayerType::kEmb);
    EXPECT_TRUE(layer.hasWeight());
    EXPECT_EQ(layer.inputCount(), 1U);
    EXPECT_EQ(layer.outputCount(), 1U);
    EXPECT_EQ(layer.weightCount(), 1U);
    EXPECT_EQ(layer.vocabSize(), kVocabSize);

    layer.setInput(0, input);
    layer.setWeight(0, weight);
    layer.setOutput(0, output);

    EXPECT_EQ(layer.forward(), eCEL::Status::kOk);
    expectEmbeddingOutput(output);
}

TEST(test_layer, embedding_layer_missing_weight_returns_invalid_argument) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, kTokenNum);
    Tensor<float> output(DeviceType::kCpu, kTokenNum, kEmbDim);

    eCEL::EmbeddingLayer layer(DeviceType::kCpu, kVocabSize);
    layer.setInput(0, input);
    layer.setOutput(0, output);

    EXPECT_EQ(layer.forward(), eCEL::Status::kInvalidArgument);
}

TEST(test_layer, rmsnorm_layer_forward_cpu) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, kRmsCount);
    Tensor<float> weight(DeviceType::kCpu, kRmsCount);
    Tensor<float> output(DeviceType::kCpu, kRmsCount);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (std::size_t i = 0; i < kRmsCount; ++i) {
        input[static_cast<int>(i)] = dist(rng);
        weight[static_cast<int>(i)] = dist(rng);
    }

    eCEL::RmsNormLayer layer(DeviceType::kCpu);
    EXPECT_EQ(layer.type(), eCEL::LayerType::kRms);
    EXPECT_TRUE(layer.hasWeight());
    EXPECT_EQ(layer.inputCount(), 1U);
    EXPECT_EQ(layer.outputCount(), 1U);
    EXPECT_EQ(layer.weightCount(), 1U);

    layer.setInput(0, input);
    layer.setWeight(0, weight);
    layer.setOutput(0, output);

    EXPECT_EQ(layer.forward(), eCEL::Status::kOk);

    float meanSquare = 0.f;
    for (std::size_t i = 0; i < kRmsCount; ++i) {
        meanSquare += input[static_cast<int>(i)] * input[static_cast<int>(i)];
    }
    meanSquare /= static_cast<float>(kRmsCount);
#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    const float eps = 1e-6f;
#else
    const float eps = 1e-5f;
#endif
    const float invRms = 1.f / std::sqrt(meanSquare + eps);
    for (std::size_t i = 0; i < kRmsCount; ++i) {
        const float expected =
            weight[static_cast<int>(i)] * input[static_cast<int>(i)] * invRms;
        ASSERT_NEAR(output[static_cast<int>(i)], expected, 1e-5f);
    }
}

TEST(test_layer, rmsnorm_layer_missing_weight_returns_invalid_argument) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, kRmsCount);
    Tensor<float> output(DeviceType::kCpu, kRmsCount);

    eCEL::RmsNormLayer layer(DeviceType::kCpu);
    layer.setInput(0, input);
    layer.setOutput(0, output);

    EXPECT_EQ(layer.forward(), eCEL::Status::kInvalidArgument);
}

TEST(test_layer, matmult_layer_forward_cpu) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, 3, 2);
    Tensor<float> weight(DeviceType::kCpu, 2, 3);
    Tensor<float> output(DeviceType::kCpu, 2, 2);
    fillMatmulExample(input, weight);

    eCEL::MatmultLayer layer(DeviceType::kCpu, 0.5f);
    EXPECT_EQ(layer.type(), eCEL::LayerType::kMatmul);
    EXPECT_TRUE(layer.hasWeight());
    EXPECT_EQ(layer.inputCount(), 1U);
    EXPECT_EQ(layer.outputCount(), 1U);
    EXPECT_EQ(layer.weightCount(), 1U);
    EXPECT_FLOAT_EQ(layer.scale(), 0.5f);

    layer.setInput(0, input);
    layer.setWeight(0, weight);
    layer.setOutput(0, output);

    EXPECT_EQ(layer.forward(), eCEL::Status::kOk);
    EXPECT_NEAR(output[0], 29.f, 1e-5f);
    EXPECT_NEAR(output[1], 69.5f, 1e-5f);
    EXPECT_NEAR(output[2], 32.f, 1e-5f);
    EXPECT_NEAR(output[3], 77.f, 1e-5f);
}

TEST(test_layer, matmult_layer_missing_weight_returns_invalid_argument) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, 3, 2);
    Tensor<float> output(DeviceType::kCpu, 2, 2);

    eCEL::MatmultLayer layer(DeviceType::kCpu);
    layer.setInput(0, input);
    layer.setOutput(0, output);

    EXPECT_EQ(layer.forward(), eCEL::Status::kInvalidArgument);
}
