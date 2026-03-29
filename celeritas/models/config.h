#pragma once

#include <cstdint>

namespace eCEL {
enum class CheckpointFormat {
    kLegacy = 0,
    kVersion1 = 1,
};

struct ModelConfig {
    int32_t dim = 0;
    int32_t hiddenDim = 0;
    int32_t layerNum = 0;
    int32_t headNum = 0;
    int32_t kvHeadNum = 0;
    int32_t vocabSize = 0;
    int32_t seqLen = 0;
#ifdef QWEN3_SUPPORT
    int32_t immediateDim = 0;
#endif
};

struct TransformerConfig {
    int32_t kvDim = 0;
    int32_t kvMul = 0;
    int32_t headSize = 0;
    int32_t vocabSize = 0;

    int32_t dim = 0;
    int32_t hiddenDim = 0;
    int32_t layerNum = 0;
    int32_t headNum = 0;
    int32_t kvHeadNum = 0;
    int32_t seqLen = 0;
    bool isSharedWeight = false;
    CheckpointFormat checkpointFormat = CheckpointFormat::kLegacy;
#ifdef QWEN3_SUPPORT
    int32_t immediateDim = 0;
#endif
};

}  // namespace eCEL
