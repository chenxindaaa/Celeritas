#include <gtest/gtest.h>
#include <cmath>

#include "celeritas/operation/AddLayer.h"
#include "celeritas/operation/EmbLayer.h"
#include "celeritas/operation/MatmulLayer.h"
#include "celeritas/operation/MlpLayer.h"
#include "celeritas/operation/RmsnormLayer.h"
#include "celeritas/operation/SwigluLayer.h"
#include "udm/core/Tensor.h"

namespace {

void fillMatmulLayerInputs(eUTIL::Tensor<float>& input, eUTIL::Tensor<float>& weight) {
    const float input_data[] = {1.f, 4.f, 2.f, 5.f, 3.f, 6.f};
    const float weight_data[] = {7.f, 9.f, 11.f, 8.f, 10.f, 12.f};

    for (int i = 0; i < 6; ++i) {
        input[i] = input_data[i];
        weight[i] = weight_data[i];
    }
}

void fillRmsnormLayerInputs(eUTIL::Tensor<float>& input, eUTIL::Tensor<float>& weight) {
    const float input_data[] = {1.f, 2.f, 3.f, 4.f};
    const float weight_data[] = {0.5f, 1.5f, 2.0f, 0.25f};

    for (int i = 0; i < 4; ++i) {
        input[i] = input_data[i];
        weight[i] = weight_data[i];
    }
}

void fillAddLayerInputs(eUTIL::Tensor<float>& input, eUTIL::Tensor<float>& value) {
    const float input_data[] = {1.f, 2.f, 3.f, 4.f};
    const float value_data[] = {0.5f, 1.5f, -2.f, 0.25f};

    for (int i = 0; i < 4; ++i) {
        input[i] = input_data[i];
        value[i] = value_data[i];
    }
}

void fillEmbLayerInputs(eUTIL::Tensor<float>& input, eUTIL::Tensor<float>& weight) {
    input[0] = 2.f;
    input[1] = 0.f;
    input[2] = 1.f;

    const float weight_data[] = {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 12.f
    };

    for (int i = 0; i < 12; ++i) {
        weight[i] = weight_data[i];
    }
}

float swigluLayerReference(float input1, float input2) {
    const float sigmoid = 1.0f / (1.0f + std::exp(-input1));
    return input1 * sigmoid * input2;
}

void fillSwigluLayerInputs(eUTIL::Tensor<float>& input1, eUTIL::Tensor<float>& input2) {
    const float input1_data[] = {-3.0f, -1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 3.0f, 6.0f};
    const float input2_data[] = {2.0f, -4.0f, 1.5f, 3.0f, -2.0f, 0.5f, 1.0f, -1.0f};

    for (int i = 0; i < 8; ++i) {
        input1[i] = input1_data[i];
        input2[i] = input2_data[i];
    }
}

float siluReference(float x) {
    return x / (1.0f + std::exp(-x));
}

}  // namespace

TEST(test_layer, matmul_layer_forward_cpu_matches_reference) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, 3, 2);
    Tensor<float> weight(DeviceType::kCpu, 2, 3);
    Tensor<float> scaler(DeviceType::kCpu);
    Tensor<float> output(DeviceType::kCpu, 2, 2);
    fillMatmulLayerInputs(input, weight);

    eCEL::Parameter<float> weight_param(weight, "weight");
    eCEL::Parameter<float> scaler_param(scaler, "scaler");
    eCEL::MatmulLayer<float, float> layer(DeviceType::kCpu,
                                          std::move(weight_param),
                                          std::move(scaler_param),
                                          1);

    eCEL::ForwardContext ctx;
    layer.forward(ctx, input, output);

    const float expected[] = {58.f, 139.f, 64.f, 154.f};
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(output[i], expected[i], 1e-5f);
    }
}

TEST(test_layer, rmsnorm_layer_forward_cpu_matches_reference) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, 4);
    Tensor<float> weight(DeviceType::kCpu, 4);
    Tensor<float> output(DeviceType::kCpu, 4);
    fillRmsnormLayerInputs(input, weight);

    eCEL::Parameter<float> weight_param(weight, "weight");
    eCEL::RmsnormLayer<float, float> layer(DeviceType::kCpu, std::move(weight_param));

    eCEL::ForwardContext ctx;
    layer.forward(ctx, input, output);

    float mean_square = 0.f;
    for (int i = 0; i < 4; ++i) {
        mean_square += input[i] * input[i];
    }
    mean_square /= 4.f;
#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    const float eps = 1e-6f;
#else
    const float eps = 1e-5f;
#endif
    const float inv_rms = 1.f / std::sqrt(mean_square + eps);

    for (int i = 0; i < 4; ++i) {
        const float expected = weight[i] * input[i] * inv_rms;
        ASSERT_NEAR(output[i], expected, 1e-5f);
    }
}

TEST(test_layer, add_layer_forward_cpu_matches_reference) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, 4);
    Tensor<float> value(DeviceType::kCpu, 4);
    Tensor<float> output(DeviceType::kCpu, 4);
    fillAddLayerInputs(input, value);

    eCEL::Parameter<float> value_param(value, "value");
    eCEL::AddLayer<float> layer(DeviceType::kCpu, std::move(value_param));

    eCEL::ForwardContext ctx;
    layer.forward(ctx, input, output);

    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(output[i], input[i] + value[i], 1e-5f);
    }
}

TEST(test_layer, emb_layer_forward_cpu_matches_reference) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    constexpr int32_t kVocabSize = 3;
    constexpr int32_t kEmbDim = 4;
    constexpr int32_t kTokenNum = 3;

    Tensor<float> input(DeviceType::kCpu, kTokenNum);
    Tensor<float> weight(DeviceType::kCpu, kVocabSize, kEmbDim);
    Tensor<float> output(DeviceType::kCpu, kTokenNum, kEmbDim);
    fillEmbLayerInputs(input, weight);

    eCEL::Parameter<float> weight_param(weight, "weight");
    eCEL::EmbLayer<float, float> layer(DeviceType::kCpu, std::move(weight_param), kVocabSize);

    eCEL::ForwardContext ctx;
    layer.forward(ctx, input, output);

    const float expected[] = {
        9.f, 10.f, 11.f, 12.f,
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f
    };

    for (int i = 0; i < kTokenNum * kEmbDim; ++i) {
        ASSERT_NEAR(output[i], expected[i], 1e-5f);
    }
}

TEST(test_layer, swiglu_layer_forward_cpu_matches_reference) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input1(DeviceType::kCpu, 8);
    Tensor<float> input2(DeviceType::kCpu, 8);
    Tensor<float> output(DeviceType::kCpu, 8);
    fillSwigluLayerInputs(input1, input2);

    eCEL::Parameter<float> value_param(input2, "value");
    eCEL::SwigluLayer<float> layer(DeviceType::kCpu, std::move(value_param));

    eCEL::ForwardContext ctx;
    layer.forward(ctx, input1, output);

    for (int i = 0; i < 8; ++i) {
        ASSERT_NEAR(output[i], swigluLayerReference(input1[i], input2[i]), 1e-5f);
    }
}

TEST(test_layer, mlp_layer_forward_cpu_matches_reference) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> input(DeviceType::kCpu, 2);
    input[0] = 1.0f;
    input[1] = -2.0f;

    Tensor<float> gate_weight(DeviceType::kCpu, 2, 2);
    Tensor<float> up_weight(DeviceType::kCpu, 2, 2);
    Tensor<float> down_weight(DeviceType::kCpu, 2, 2);
    Tensor<float> empty_scaler(DeviceType::kCpu);
    Tensor<float> output(DeviceType::kCpu, 2);

    gate_weight[0] = 1.0f;  gate_weight[1] = 0.0f;
    gate_weight[2] = 0.0f;  gate_weight[3] = 1.0f;

    up_weight[0] = 2.0f;    up_weight[1] = 0.0f;
    up_weight[2] = 0.0f;    up_weight[3] = -1.0f;

    down_weight[0] = 1.0f;  down_weight[1] = 0.0f;
    down_weight[2] = 0.0f;  down_weight[3] = 1.0f;

    eCEL::MlpParams<float> params(
        eCEL::MatmulParams<float>(
            eCEL::Parameter<float>(gate_weight, "w1"),
            eCEL::Parameter<float>(empty_scaler, "w1_scale"),
            1),
        eCEL::MatmulParams<float>(
            eCEL::Parameter<float>(down_weight, "w2"),
            eCEL::Parameter<float>(empty_scaler, "w2_scale"),
            1),
        eCEL::MatmulParams<float>(
            eCEL::Parameter<float>(up_weight, "w3"),
            eCEL::Parameter<float>(empty_scaler, "w3_scale"),
            1));
    eCEL::MlpLayer<float, float> layer(DeviceType::kCpu, std::move(params));

    eCEL::ForwardContext ctx;
    layer.forward(ctx, input, output);

    const float gate0 = 1.0f;
    const float gate1 = -2.0f;
    const float up0 = 2.0f;
    const float up1 = 2.0f;
    const float hidden0 = siluReference(gate0) * up0;
    const float hidden1 = siluReference(gate1) * up1;
    const float expected0 = input[0] + hidden0;
    const float expected1 = input[1] + hidden1;

    ASSERT_NEAR(output[0], expected0, 1e-5f);
    ASSERT_NEAR(output[1], expected1, 1e-5f);
}
