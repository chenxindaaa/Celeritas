#include <gtest/gtest.h>
#include <cmath>

#include "celeritas/kvCache/kvCacheMgr.h"
#include "celeritas/operation/AddLayer.h"
#include "celeritas/operation/DecoderLayer.h"
#include "celeritas/operation/EmbLayer.h"
#include "celeritas/operation/MatmulLayer.h"
#include "celeritas/operation/MhaLayer.h"
#include "celeritas/operation/MlpLayer.h"
#include "celeritas/operation/RmsnormLayer.h"
#include "celeritas/ropeCache/RopeCache.h"
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

std::vector<float> computeMhaHeadScores(const float* query,
                                        const float* keyBase,
                                        int32_t pos,
                                        int32_t kvDim,
                                        int32_t headSize) {
    std::vector<float> scores(static_cast<std::size_t>(pos + 1), 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(headSize));
    for (int32_t t = 0; t <= pos; ++t) {
        float dot = 0.0f;
        const float* key = keyBase + t * kvDim;
        for (int32_t i = 0; i < headSize; ++i) {
            dot += query[i] * key[i];
        }
        scores[static_cast<std::size_t>(t)] = dot * scale;
    }
    return scores;
}

void softmaxInplace(std::vector<float>& values) {
    float maxValue = values[0];
    for (float value : values) {
        maxValue = std::max(maxValue, value);
    }

    float sum = 0.0f;
    for (float& value : values) {
        value = std::exp(value - maxValue);
        sum += value;
    }

    for (float& value : values) {
        value /= sum;
    }
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

    eCEL::Parameter<float> weight_param(weight);
    eCEL::Parameter<float> scaler_param(scaler);
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

    eCEL::Parameter<float> weight_param(weight);
    eCEL::RmsnormLayer<float> layer(DeviceType::kCpu, std::move(weight_param));

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

    eCEL::AddLayer<float> layer(DeviceType::kCpu);

    eCEL::ForwardContext ctx;
    const Tensor<float>* addInputs[] = {&input, &value};
    layer.forward(ctx, eCEL::TensorListView<float>(addInputs, 2), output);

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

    eCEL::Parameter<float> weight_param(weight);
    eCEL::EmbLayer<float> layer(DeviceType::kCpu, std::move(weight_param), kVocabSize);

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

    eCEL::SwigluLayer<float> layer(DeviceType::kCpu);

    eCEL::ForwardContext ctx;
    const Tensor<float>* swigluInputs[] = {&input1, &input2};
    layer.forward(ctx, eCEL::TensorListView<float>(swigluInputs, 2), output);

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

    eCEL::MlpParams<float> params{
        eCEL::MatmulParams<float>(
            eCEL::Parameter<float>(gate_weight),
            eCEL::Parameter<float>(empty_scaler)),
        eCEL::MatmulParams<float>(
            eCEL::Parameter<float>(down_weight),
            eCEL::Parameter<float>(empty_scaler)),
        eCEL::MatmulParams<float>(
            eCEL::Parameter<float>(up_weight),
            eCEL::Parameter<float>(empty_scaler))};
    eCEL::MlpLayer<float, float> layer(DeviceType::kCpu, std::move(params));

    eCEL::ForwardContext ctx;
    layer.forward(ctx, input, output);

    const float gate0 = 1.0f;
    const float gate1 = -2.0f;
    const float up0 = 2.0f;
    const float up1 = 2.0f;
    const float hidden0 = siluReference(gate0) * up0;
    const float hidden1 = siluReference(gate1) * up1;
    const float expected0 = hidden0;
    const float expected1 = hidden1;

    ASSERT_NEAR(output[0], expected0, 1e-5f);
    ASSERT_NEAR(output[1], expected1, 1e-5f);
}

TEST(test_layer, mha_layer_forward_cpu_matches_reference) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    constexpr int32_t pos = 1;
    constexpr int32_t headNum = 2;
    constexpr int32_t seqLen = 3;
    constexpr int32_t headSize = 2;
    constexpr int32_t kvMul = 1;
    constexpr int32_t kvDim = headNum * headSize;

    Tensor<float> query(DeviceType::kCpu, headNum, headSize);
    Tensor<float> keyCache(DeviceType::kCpu, seqLen, kvDim);
    Tensor<float> valueCache(DeviceType::kCpu, seqLen, kvDim);
    Tensor<float> output(DeviceType::kCpu, headNum, headSize);

    const float queryData[] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const float keyData[] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    const float valueData[] = {
        10.0f, 20.0f, 30.0f, 40.0f,
        50.0f, 60.0f, 70.0f, 80.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };

    for (int i = 0; i < headNum * headSize; ++i) {
        query[i] = queryData[i];
        output[i] = 0.0f;
    }
    for (int i = 0; i < seqLen * kvDim; ++i) {
        keyCache[i] = keyData[i];
        valueCache[i] = valueData[i];
    }

    eCEL::KVCacheView cacheView;
    cacheView.k_ptr = keyCache.data();
    cacheView.v_ptr = valueCache.data();
    cacheView.seq_len = static_cast<std::size_t>(pos + 1);

    eCEL::MhaLayer<float> layer(
        DeviceType::kCpu,
        headNum,
        seqLen,
        kvDim,
        kvMul,
        headSize);

    eCEL::ForwardContext ctx;
    ctx.kv_len = pos + 1;
    ctx.kv_cache = &cacheView;
    layer.forward(ctx, query, output);

    for (int32_t h = 0; h < headNum; ++h) {
        const float* queryHead = query.data() + h * headSize;
        const float* keyHeadBase = keyCache.data() + h * headSize;
        const float* valueHeadBase = valueCache.data() + h * headSize;
        std::vector<float> scores = computeMhaHeadScores(queryHead, keyHeadBase, pos, kvDim, headSize);
        softmaxInplace(scores);

        for (int32_t i = 0; i < headSize; ++i) {
            float expected = 0.0f;
            for (int32_t t = 0; t <= pos; ++t) {
                expected += scores[static_cast<std::size_t>(t)] *
                            valueHeadBase[t * kvDim + i];
            }
            ASSERT_NEAR(output[h * headSize + i], expected, 1e-5f);
        }
    }
}

TEST(test_layer, decoder_layer_forward_cpu_matches_reference) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    constexpr int32_t headNum = 1;
    constexpr int32_t seqLen = 2;
    constexpr int32_t headSize = 2;
    constexpr int32_t kvMul = 1;
    constexpr int32_t kvDim = 2;

    Tensor<float> input(DeviceType::kCpu, 2);
    input[0] = 1.0f;
    input[1] = -2.0f;

    Tensor<float> attnNormWeight(DeviceType::kCpu, 2);
    attnNormWeight[0] = 1.0f;
    attnNormWeight[1] = 1.0f;
    Tensor<float> ffnNormWeight(DeviceType::kCpu, 2);
    ffnNormWeight[0] = 1.0f;
    ffnNormWeight[1] = 1.0f;

    Tensor<float> gateWeight(DeviceType::kCpu, 2, 2);
    Tensor<float> upWeight(DeviceType::kCpu, 2, 2);
    Tensor<float> downWeight(DeviceType::kCpu, 2, 2);
    Tensor<float> wqWeight(DeviceType::kCpu, 2, 2);
    Tensor<float> wkWeight(DeviceType::kCpu, 2, 2);
    Tensor<float> wvWeight(DeviceType::kCpu, 2, 2);
    Tensor<float> woWeight(DeviceType::kCpu, 2, 2);
    Tensor<float> emptyScaler(DeviceType::kCpu);
    Tensor<float> keyCache(DeviceType::kCpu, seqLen, kvDim);
    Tensor<float> valueCache(DeviceType::kCpu, seqLen, kvDim);
    Tensor<float> sinCache(DeviceType::kCpu, seqLen, headSize);
    Tensor<float> cosCache(DeviceType::kCpu, seqLen, headSize);
    Tensor<float> output(DeviceType::kCpu, 2);

    for (int i = 0; i < 4; ++i) {
        gateWeight[i] = 0.0f;
        upWeight[i] = 0.0f;
        downWeight[i] = 0.0f;
        wqWeight[i] = 0.0f;
        wkWeight[i] = 0.0f;
        wvWeight[i] = 0.0f;
        woWeight[i] = 0.0f;
    }
    for (int i = 0; i < seqLen * headSize; ++i) {
        sinCache[i] = 0.0f;
        cosCache[i] = 1.0f;
    }

    const float keyData[] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const float valueData[] = {
        0.0f, 0.0f,
        0.0f, 0.0f
    };
    for (int i = 0; i < seqLen * kvDim; ++i) {
        keyCache[i] = keyData[i];
        valueCache[i] = valueData[i];
    }

    eCEL::KVCacheView cacheView;
    cacheView.k_ptr = keyCache.data();
    cacheView.v_ptr = valueCache.data();
    cacheView.seq_len = 0;

    eCEL::RopeCacheView ropeCacheView;
    ropeCacheView.sin_ptr = sinCache.data();
    ropeCacheView.cos_ptr = cosCache.data();
    ropeCacheView.seq_len = seqLen;
    ropeCacheView.head_size = headSize;

    eCEL::DecoderParams<float> params{
        eCEL::RmsnormParams<float>(eCEL::Parameter<float>(attnNormWeight)),
        eCEL::SelfAttentionParams<float>(
            eCEL::MatmulParams<float>(
                eCEL::Parameter<float>(wqWeight),
                eCEL::Parameter<float>(emptyScaler)),
            eCEL::MatmulParams<float>(
                eCEL::Parameter<float>(wkWeight),
                eCEL::Parameter<float>(emptyScaler)),
            eCEL::MatmulParams<float>(
                eCEL::Parameter<float>(wvWeight),
                eCEL::Parameter<float>(emptyScaler)),
            eCEL::MatmulParams<float>(
                eCEL::Parameter<float>(woWeight),
                eCEL::Parameter<float>(emptyScaler))),
        eCEL::RmsnormParams<float>(eCEL::Parameter<float>(ffnNormWeight)),
        eCEL::MlpParams<float>(
            eCEL::MatmulParams<float>(
                eCEL::Parameter<float>(gateWeight),
                eCEL::Parameter<float>(emptyScaler)),
            eCEL::MatmulParams<float>(
                eCEL::Parameter<float>(downWeight),
                eCEL::Parameter<float>(emptyScaler)),
            eCEL::MatmulParams<float>(
                eCEL::Parameter<float>(upWeight),
                eCEL::Parameter<float>(emptyScaler)))};
    eCEL::DecoderLayer<float, float> layer(DeviceType::kCpu, std::move(params));
    layer.setMhaConfig(headNum, seqLen, kvDim, kvMul, headSize);

    eCEL::ForwardContext ctx;
    ctx.kv_len = 1;
    ctx.kv_cache = &cacheView;
    ctx.rope_cache = &ropeCacheView;
    layer.forward(ctx, input, output);

    ASSERT_NEAR(output[0], input[0], 1e-5f);
    ASSERT_NEAR(output[1], input[1], 1e-5f);
}
