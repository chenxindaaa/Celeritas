#include <cmath>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "celeritas/kernels/KernelFactory.h"
#include "udm/common/cudaConfig.h"
#include "udm/core/Tensor.h"

namespace {

struct MhaReference {
    std::vector<float> scores;
    std::vector<float> output;
};

std::vector<float> computeHeadScores(const float* query,
                                     const float* keyBase,
                                     int32_t pos,
                                     int32_t seqLen,
                                     int32_t kvDim,
                                     int32_t headSize) {
    (void)seqLen;

    std::vector<float> scores(pos + 1, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(headSize));
    for (int32_t t = 0; t <= pos; ++t) {
        float dot = 0.0f;
        const float* key = keyBase + t * kvDim;
        for (int32_t i = 0; i < headSize; ++i) {
            dot += query[i] * key[i];
        }
        scores[t] = dot * scale;
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

MhaReference computeMhaReference(int32_t pos,
                                 int32_t headNum,
                                 int32_t layerIndex,
                                 int32_t seqLen,
                                 int32_t kvDim,
                                 int32_t kvMul,
                                 int32_t headSize,
                                 const eUTIL::Tensor<float>& query,
                                 const eUTIL::Tensor<float>& keyCache,
                                 const eUTIL::Tensor<float>& valueCache) {
    MhaReference reference;
    reference.scores.assign(headNum * seqLen, 0.0f);
    reference.output.assign(headNum * headSize, 0.0f);
    const int32_t layerOffset = layerIndex * seqLen * kvDim;

    for (int32_t h = 0; h < headNum; ++h) {
        const float* queryHead = query.data() + h * headSize;
        const float* keyHeadBase = keyCache.data() + layerOffset + (h / kvMul) * headSize;
        const float* valueHeadBase = valueCache.data() + layerOffset + (h / kvMul) * headSize;

        std::vector<float> scores = computeHeadScores(queryHead, keyHeadBase, pos, seqLen, kvDim, headSize);
        softmaxInplace(scores);
        for (int32_t t = 0; t <= pos; ++t) {
            reference.scores[h * seqLen + t] = scores[t];
        }

        for (int32_t i = 0; i < headSize; ++i) {
            float acc = 0.0f;
            for (int32_t t = 0; t <= pos; ++t) {
                acc += scores[t] * valueHeadBase[t * kvDim + i];
            }
            reference.output[h * headSize + i] = acc;
        }
    }

    return reference;
}

void expectTensorNear(const eUTIL::Tensor<float>& actual,
                      const std::vector<float>& expected,
                      float tol = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        ASSERT_NEAR(actual.data()[i], expected[i], tol) << "mismatch at index " << i;
    }
}

}  // namespace

TEST(test_mha, mha_cpu_matches_reference_for_multiple_heads) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    constexpr int32_t pos = 1;
    constexpr int32_t headNum = 2;
    constexpr int32_t layerIndex = 0;
    constexpr int32_t seqLen = 3;
    constexpr int32_t headSize = 2;
    constexpr int32_t kvMul = 1;
    constexpr int32_t kvDim = headNum * headSize;

    Tensor<float> query(DeviceType::kCpu, headNum, headSize);
    Tensor<float> keyCache(DeviceType::kCpu, seqLen, kvDim);
    Tensor<float> valueCache(DeviceType::kCpu, seqLen, kvDim);
    Tensor<float> score(DeviceType::kCpu, headNum, seqLen);
    Tensor<float> output(DeviceType::kCpu, headNum, headSize);

    const float queryData[] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const float keyData[] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 0.0f,
        -3.0f, -3.0f, -3.0f, -3.0f
    };
    const float valueData[] = {
        10.0f, 20.0f, 30.0f, 40.0f,
        50.0f, 60.0f, 70.0f, 80.0f,
        99.0f, 99.0f, 99.0f, 99.0f
    };

    for (int i = 0; i < headNum * headSize; ++i) {
        query[i] = queryData[i];
        output[i] = 0.0f;
    }
    for (int i = 0; i < headNum * seqLen; ++i) {
        score[i] = 0.0f;
    }
    for (int i = 0; i < seqLen * kvDim; ++i) {
        keyCache[i] = keyData[i];
        valueCache[i] = valueData[i];
    }

    const MhaReference expected =
        computeMhaReference(pos, headNum, layerIndex, seqLen, kvDim, kvMul, headSize, query, keyCache, valueCache);

    auto kernel = eCEL::KernelFactory::getMhaKernel();
    kernel(pos, headNum, layerIndex, seqLen, kvDim, kvMul, headSize,
           query, keyCache, valueCache, score, output, nullptr);

    expectTensorNear(score, expected.scores);
    expectTensorNear(output, expected.output);
}

TEST(test_mha, mha_cpu_respects_kv_head_sharing) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    constexpr int32_t pos = 1;
    constexpr int32_t headNum = 4;
    constexpr int32_t layerIndex = 0;
    constexpr int32_t seqLen = 2;
    constexpr int32_t headSize = 1;
    constexpr int32_t kvMul = 2;
    constexpr int32_t kvDim = 2;

    Tensor<float> query(DeviceType::kCpu, headNum, headSize);
    Tensor<float> keyCache(DeviceType::kCpu, seqLen, kvDim);
    Tensor<float> valueCache(DeviceType::kCpu, seqLen, kvDim);
    Tensor<float> score(DeviceType::kCpu, headNum, seqLen);
    Tensor<float> output(DeviceType::kCpu, headNum, headSize);

    const float queryData[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float keyData[] = {
        1.0f, 3.0f,
        2.0f, 4.0f
    };
    const float valueData[] = {
        100.0f, 1000.0f,
        200.0f, 2000.0f
    };

    for (int i = 0; i < headNum; ++i) {
        query[i] = queryData[i];
        output[i] = 0.0f;
    }
    for (int i = 0; i < headNum * seqLen; ++i) {
        score[i] = 0.0f;
    }
    for (int i = 0; i < seqLen * kvDim; ++i) {
        keyCache[i] = keyData[i];
        valueCache[i] = valueData[i];
    }

    const MhaReference expected =
        computeMhaReference(pos, headNum, layerIndex, seqLen, kvDim, kvMul, headSize, query, keyCache, valueCache);

    auto kernel = eCEL::KernelFactory::getMhaKernel();
    kernel(pos, headNum, layerIndex, seqLen, kvDim, kvMul, headSize,
           query, keyCache, valueCache, score, output, nullptr);

    expectTensorNear(score, expected.scores);
    expectTensorNear(output, expected.output);
}

TEST(test_mha, mha_gpu_matches_reference_for_multiple_heads) {
    using eUTIL::CudaConfig;
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        GTEST_SKIP() << "CUDA device not available";
    }

    constexpr int32_t pos = 1;
    constexpr int32_t headNum = 2;
    constexpr int32_t layerIndex = 0;
    constexpr int32_t seqLen = 3;
    constexpr int32_t headSize = 4;
    constexpr int32_t kvMul = 1;
    constexpr int32_t kvDim = headNum * headSize;

    Tensor<float> cpuQuery(DeviceType::kCpu, headNum, headSize);
    Tensor<float> cpuKeyCache(DeviceType::kCpu, seqLen, kvDim);
    Tensor<float> cpuValueCache(DeviceType::kCpu, seqLen, kvDim);
    Tensor<float> cpuScore(DeviceType::kCpu, headNum, seqLen);
    Tensor<float> cpuOutput(DeviceType::kCpu, headNum, headSize);

    const float queryData[] = {
        1.0f, 0.0f, 2.0f, -1.0f,
        0.5f, 1.5f, -0.5f, 2.0f
    };
    const float keyData[] = {
        1.0f, 0.0f, 1.0f, 0.0f,   0.0f, 1.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 1.0f,   1.0f, 0.0f, 1.0f, 0.0f,
        -9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f
    };
    const float valueData[] = {
        10.0f, 20.0f, 30.0f, 40.0f,   50.0f, 60.0f, 70.0f, 80.0f,
        15.0f, 25.0f, 35.0f, 45.0f,   55.0f, 65.0f, 75.0f, 85.0f,
        99.0f, 99.0f, 99.0f, 99.0f,   99.0f, 99.0f, 99.0f, 99.0f
    };

    for (int i = 0; i < headNum * headSize; ++i) {
        cpuQuery[i] = queryData[i];
        cpuOutput[i] = 0.0f;
    }
    for (int i = 0; i < headNum * seqLen; ++i) {
        cpuScore[i] = 0.0f;
    }
    for (int i = 0; i < seqLen * kvDim; ++i) {
        cpuKeyCache[i] = keyData[i];
        cpuValueCache[i] = valueData[i];
    }

    const MhaReference expected = computeMhaReference(
        pos, headNum, layerIndex, seqLen, kvDim, kvMul, headSize, cpuQuery, cpuKeyCache, cpuValueCache);

    Tensor<float> gpuQuery = cpuQuery.cuda();
    Tensor<float> gpuKeyCache = cpuKeyCache.cuda();
    Tensor<float> gpuValueCache = cpuValueCache.cuda();
    Tensor<float> gpuScore(DeviceType::kCuda, headNum, seqLen);
    Tensor<float> gpuOutput(DeviceType::kCuda, headNum, headSize);

    CudaConfig config;
    auto kernel = eCEL::KernelFactory::getMhaKernel();
    kernel(pos, headNum, layerIndex, seqLen, kvDim, kvMul, headSize,
           gpuQuery, gpuKeyCache, gpuValueCache, gpuScore, gpuOutput, &config);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    gpuScore.cpu();
    gpuOutput.cpu();

    expectTensorNear(gpuScore, expected.scores, 1e-4f);
    expectTensorNear(gpuOutput, expected.output, 1e-4f);
}

TEST(test_mha, mha_gpu_respects_kv_head_sharing) {
    using eUTIL::CudaConfig;
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        GTEST_SKIP() << "CUDA device not available";
    }

    constexpr int32_t pos = 1;
    constexpr int32_t headNum = 4;
    constexpr int32_t layerIndex = 0;
    constexpr int32_t seqLen = 2;
    constexpr int32_t headSize = 4;
    constexpr int32_t kvMul = 2;
    constexpr int32_t kvDim = 8;

    Tensor<float> cpuQuery(DeviceType::kCpu, headNum, headSize);
    Tensor<float> cpuKeyCache(DeviceType::kCpu, seqLen, kvDim);
    Tensor<float> cpuValueCache(DeviceType::kCpu, seqLen, kvDim);
    Tensor<float> cpuScore(DeviceType::kCpu, headNum, seqLen);
    Tensor<float> cpuOutput(DeviceType::kCpu, headNum, headSize);

    const float queryData[] = {
        1.0f, 0.0f, 0.5f, -0.5f,
        0.0f, 1.0f, 1.5f, 0.5f,
        2.0f, -1.0f, 0.0f, 1.0f,
        1.0f, 1.0f, -1.0f, 2.0f
    };
    const float keyData[] = {
        1.0f, 0.0f, 1.0f, 0.0f,   0.0f, 1.0f, 0.0f, 1.0f,
        0.5f, 1.0f, 0.0f, 1.0f,   1.0f, 0.5f, 1.0f, 0.0f
    };
    const float valueData[] = {
        100.0f, 110.0f, 120.0f, 130.0f,   200.0f, 210.0f, 220.0f, 230.0f,
        300.0f, 310.0f, 320.0f, 330.0f,   400.0f, 410.0f, 420.0f, 430.0f
    };

    for (int i = 0; i < headNum * headSize; ++i) {
        cpuQuery[i] = queryData[i];
        cpuOutput[i] = 0.0f;
    }
    for (int i = 0; i < headNum * seqLen; ++i) {
        cpuScore[i] = 0.0f;
    }
    for (int i = 0; i < seqLen * kvDim; ++i) {
        cpuKeyCache[i] = keyData[i];
        cpuValueCache[i] = valueData[i];
    }

    const MhaReference expected = computeMhaReference(
        pos, headNum, layerIndex, seqLen, kvDim, kvMul, headSize, cpuQuery, cpuKeyCache, cpuValueCache);

    Tensor<float> gpuQuery = cpuQuery.cuda();
    Tensor<float> gpuKeyCache = cpuKeyCache.cuda();
    Tensor<float> gpuValueCache = cpuValueCache.cuda();
    Tensor<float> gpuScore(DeviceType::kCuda, headNum, seqLen);
    Tensor<float> gpuOutput(DeviceType::kCuda, headNum, headSize);

    CudaConfig config;
    auto kernel = eCEL::KernelFactory::getMhaKernel();
    kernel(pos, headNum, layerIndex, seqLen, kvDim, kvMul, headSize,
           gpuQuery, gpuKeyCache, gpuValueCache, gpuScore, gpuOutput, &config);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    gpuScore.cpu();
    gpuOutput.cpu();

    expectTensorNear(gpuScore, expected.scores, 1e-4f);
    expectTensorNear(gpuOutput, expected.output, 1e-4f);
}

// TEST(test_mha, mha_gpu_matches_reference_when_head_size_is_not_multiple_of_four) {
//     using eUTIL::CudaConfig;
//     using eUTIL::DeviceType;
//     using eUTIL::Tensor;

//     int deviceCount = 0;
//     if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
//         GTEST_SKIP() << "CUDA device not available";
//     }

//     constexpr int32_t pos = 2;
//     constexpr int32_t headNum = 8;
//     constexpr int32_t layerIndex = 0;
//     constexpr int32_t seqLen = 3;
//     constexpr int32_t headSize = 5;
//     constexpr int32_t kvMul = 1;
//     constexpr int32_t kvDim = headNum * headSize;

//     Tensor<float> cpuQuery(DeviceType::kCpu, headNum, headSize);
//     Tensor<float> cpuKeyCache(DeviceType::kCpu, seqLen, kvDim);
//     Tensor<float> cpuValueCache(DeviceType::kCpu, seqLen, kvDim);
//     Tensor<float> cpuScore(DeviceType::kCpu, headNum, seqLen);
//     Tensor<float> cpuOutput(DeviceType::kCpu, headNum, headSize);

//     for (int i = 0; i < headNum * headSize; ++i) {
//         const int32_t h = i / headSize;
//         const int32_t d = i % headSize;
//         cpuQuery[i] = (h % 2 == 0 ? 1.0f : -1.0f) * (10.0f * (h + 1) + static_cast<float>(d) * 0.5f);
//         cpuOutput[i] = 0.0f;
//     }
//     for (int i = 0; i < headNum * seqLen; ++i) {
//         cpuScore[i] = 0.0f;
//     }
//     for (int i = 0; i < seqLen * kvDim; ++i) {
//         const int32_t t = i / kvDim;
//         const int32_t withinToken = i % kvDim;
//         const int32_t h = withinToken / headSize;
//         const int32_t d = withinToken % headSize;

//         cpuKeyCache[i] = 50.0f * static_cast<float>(h + 1) +
//                          7.0f * static_cast<float>(t + 1) +
//                          1.25f * static_cast<float>(d + 1);
//         cpuValueCache[i] = 1000.0f * static_cast<float>(t + 1) +
//                            100.0f * static_cast<float>(h + 1) +
//                            10.0f * static_cast<float>(d + 1);
//     }

//     const MhaReference expected = computeMhaReference(
//         pos, headNum, layerIndex, seqLen, kvDim, kvMul, headSize, cpuQuery, cpuKeyCache, cpuValueCache);

//     Tensor<float> gpuQuery = cpuQuery.cuda();
//     Tensor<float> gpuKeyCache = cpuKeyCache.cuda();
//     Tensor<float> gpuValueCache = cpuValueCache.cuda();
//     Tensor<float> gpuScore(DeviceType::kCuda, headNum, seqLen);
//     Tensor<float> gpuOutput(DeviceType::kCuda, headNum, headSize);

//     CudaConfig config;
//     auto kernel = eCEL::KernelFactory::getMhaKernel();
//     kernel(pos, headNum, layerIndex, seqLen, kvDim, kvMul, headSize,
//            gpuQuery, gpuKeyCache, gpuValueCache, gpuScore, gpuOutput, &config);
//     ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

//     gpuScore.cpu();
//     gpuOutput.cpu();

//     expectTensorNear(gpuScore, expected.scores, 1e-4f);
//     expectTensorNear(gpuOutput, expected.output, 1e-4f);
// }

// TEST(test_mha, mha_gpu_matches_reference_when_head_num_is_one_and_head_size_is_not_multiple_of_four) {
//     using eUTIL::CudaConfig;
//     using eUTIL::DeviceType;
//     using eUTIL::Tensor;

//     int deviceCount = 0;
//     if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
//         GTEST_SKIP() << "CUDA device not available";
//     }

//     constexpr int32_t pos = 2;
//     constexpr int32_t headNum = 1;
//     constexpr int32_t layerIndex = 0;
//     constexpr int32_t seqLen = 3;
//     constexpr int32_t headSize = 5;
//     constexpr int32_t kvMul = 1;
//     constexpr int32_t kvDim = headNum * headSize;

//     Tensor<float> cpuQuery(DeviceType::kCpu, headNum, headSize);
//     Tensor<float> cpuKeyCache(DeviceType::kCpu, seqLen, kvDim);
//     Tensor<float> cpuValueCache(DeviceType::kCpu, seqLen, kvDim);
//     Tensor<float> cpuScore(DeviceType::kCpu, headNum, seqLen);
//     Tensor<float> cpuOutput(DeviceType::kCpu, headNum, headSize);

//     for (int i = 0; i < headNum * headSize; ++i) {
//         const int32_t h = i / headSize;
//         const int32_t d = i % headSize;
//         cpuQuery[i] = (h % 2 == 0 ? 1.0f : -1.0f) * (10.0f * (h + 1) + static_cast<float>(d) * 0.5f);
//         cpuOutput[i] = 0.0f;
//     }
//     for (int i = 0; i < headNum * seqLen; ++i) {
//         cpuScore[i] = 0.0f;
//     }
//     for (int i = 0; i < seqLen * kvDim; ++i) {
//         const int32_t t = i / kvDim;
//         const int32_t withinToken = i % kvDim;
//         const int32_t h = withinToken / headSize;
//         const int32_t d = withinToken % headSize;

//         cpuKeyCache[i] = 50.0f * static_cast<float>(h + 1) +
//                          7.0f * static_cast<float>(t + 1) +
//                          1.25f * static_cast<float>(d + 1);
//         cpuValueCache[i] = 1000.0f * static_cast<float>(t + 1) +
//                            100.0f * static_cast<float>(h + 1) +
//                            10.0f * static_cast<float>(d + 1);
//     }

//     const MhaReference expected = computeMhaReference(
//         pos, headNum, layerIndex, seqLen, kvDim, kvMul, headSize, cpuQuery, cpuKeyCache, cpuValueCache);

//     Tensor<float> gpuQuery = cpuQuery.cuda();
//     Tensor<float> gpuKeyCache = cpuKeyCache.cuda();
//     Tensor<float> gpuValueCache = cpuValueCache.cuda();
//     Tensor<float> gpuScore(DeviceType::kCuda, headNum, seqLen);
//     Tensor<float> gpuOutput(DeviceType::kCuda, headNum, headSize);

//     CudaConfig config;
//     auto kernel = eCEL::KernelFactory::getMhaKernel();
//     kernel(pos, headNum, layerIndex, seqLen, kvDim, kvMul, headSize,
//            gpuQuery, gpuKeyCache, gpuValueCache, gpuScore, gpuOutput, &config);
//     ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

//     gpuScore.cpu();
//     gpuOutput.cpu();

//     expectTensorNear(gpuScore, expected.scores, 1e-4f);
//     expectTensorNear(gpuOutput, expected.output, 1e-4f);
// }
