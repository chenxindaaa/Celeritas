#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "celeritas/engine/LLMEngine.h"
#include "celeritas/loader/Loader.h"
#include "celeritas/models/Llama2.h"
#include "celeritas/models/config.h"
#include "celeritas/tokenizer/SentencePieceTokenizer.h"

namespace {

constexpr uint32_t kCheckpointMagic = 0x616b3432;
constexpr std::streamoff kVersionedHeaderSize = 256;

struct CheckpointInfo {
    eCEL::ModelConfig config{};
    std::streamoff embeddingOffset = static_cast<std::streamoff>(sizeof(eCEL::ModelConfig));
};

std::string checkpointPath() {
    return "/home/cxd/Celeritas/models/stories110M.bin";
}

std::string tokenizerModelPath() {
    return "/home/cxd/Celeritas/models/tokenizer.model";
}

std::string tokenizerGoldenPath() {
    return "/home/cxd/Celeritas/test/testModel/sentencepiece_golden.txt";
}

struct SentencePieceGolden {
    std::string prompt;
    std::vector<eCEL::TokenId> ids;
};

SentencePieceGolden loadSentencePieceGolden() {
    const std::string goldenPath = tokenizerGoldenPath();
    if (!std::filesystem::exists(goldenPath)) {
        throw std::runtime_error("SentencePiece golden file not found: " + goldenPath);
    }

    std::ifstream input(goldenPath);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open SentencePiece golden file: " + goldenPath);
    }

    SentencePieceGolden golden;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        if (line.rfind("prompt=", 0) == 0) {
            golden.prompt = line.substr(7);
            continue;
        }

        if (line.rfind("ids=", 0) == 0) {
            std::stringstream stream(line.substr(4));
            eCEL::TokenId id = 0;
            while (stream >> id) {
                golden.ids.push_back(id);
            }
        }
    }

    if (golden.prompt.empty()) {
        throw std::runtime_error("SentencePiece golden prompt is empty");
    }
    if (golden.ids.empty()) {
        throw std::runtime_error("SentencePiece golden ids are empty");
    }

    return golden;
}

std::vector<float> loadExpectedEmbeddingRow(eCEL::TokenId tokenId, int32_t* dimOut = nullptr) {
    const std::string modelPath = checkpointPath();
    if (!std::filesystem::exists(modelPath)) {
        throw std::runtime_error("Model checkpoint not found: " + modelPath);
    }

    std::ifstream input(modelPath, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open model checkpoint: " + modelPath);
    }

    CheckpointInfo checkpointInfo;

    uint32_t magic = 0;
    input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!input) {
        throw std::runtime_error("Failed to read checkpoint header from: " + modelPath);
    }

    if (magic == kCheckpointMagic) {
        int32_t version = 0;
        input.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (!input) {
            throw std::runtime_error("Failed to read checkpoint version from: " + modelPath);
        }
        if (version != 1) {
            throw std::runtime_error("Unsupported checkpoint version in test: " + std::to_string(version));
        }

        input.read(reinterpret_cast<char*>(&checkpointInfo.config), sizeof(checkpointInfo.config));
        if (!input) {
            throw std::runtime_error("Failed to read ModelConfig from versioned checkpoint: " + modelPath);
        }

        const int32_t vocabSize = std::abs(checkpointInfo.config.vocabSize);
        const std::streamoff weightsOffset = kVersionedHeaderSize;
        checkpointInfo.embeddingOffset =
            weightsOffset +
            static_cast<std::streamoff>(checkpointInfo.config.layerNum) *
                static_cast<std::streamoff>(checkpointInfo.config.dim) *
                static_cast<std::streamoff>(sizeof(float)) * 2 +
            static_cast<std::streamoff>(checkpointInfo.config.dim) * static_cast<std::streamoff>(sizeof(float));
        checkpointInfo.embeddingOffset +=
            static_cast<std::streamoff>(tokenId) * checkpointInfo.config.dim *
                static_cast<std::streamoff>(sizeof(float));
        if (tokenId < 0 || tokenId >= vocabSize) {
            throw std::out_of_range("Token id is out of checkpoint vocab range");
        }
    } else {
        input.seekg(0);
        input.read(reinterpret_cast<char*>(&checkpointInfo.config), sizeof(checkpointInfo.config));
        if (!input) {
            throw std::runtime_error("Failed to read legacy ModelConfig from checkpoint: " + modelPath);
        }
        checkpointInfo.embeddingOffset =
            static_cast<std::streamoff>(sizeof(eCEL::ModelConfig)) +
            static_cast<std::streamoff>(tokenId) * checkpointInfo.config.dim *
                static_cast<std::streamoff>(sizeof(float));
    }

    const eCEL::ModelConfig& config = checkpointInfo.config;
    if (config.dim <= 0 || config.vocabSize == 0) {
        throw std::runtime_error("Invalid ModelConfig in checkpoint: " + modelPath);
    }

    const int32_t vocabSize = std::abs(config.vocabSize);
    if (tokenId < 0 || tokenId >= vocabSize) {
        throw std::out_of_range("Token id is out of checkpoint vocab range");
    }

    input.seekg(checkpointInfo.embeddingOffset);
    if (!input) {
        throw std::runtime_error("Failed to seek to embedding row in checkpoint: " + modelPath);
    }

    std::vector<float> row(static_cast<std::size_t>(config.dim));
    input.read(reinterpret_cast<char*>(row.data()),
               static_cast<std::streamsize>(row.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("Failed to read embedding row from checkpoint: " + modelPath);
    }

    if (dimOut != nullptr) {
        *dimOut = config.dim;
    }
    return row;
}

}  // namespace

TEST(test_llama2,
     llm_engine_with_sentencepiece_tokenizer_returns_last_token_logits) {
    ASSERT_TRUE(std::filesystem::exists(checkpointPath()));
    ASSERT_TRUE(std::filesystem::exists(tokenizerModelPath()));
    ASSERT_TRUE(std::filesystem::exists(tokenizerGoldenPath()));

    {
        std::ifstream input(checkpointPath(), std::ios::binary);
        ASSERT_TRUE(input.is_open());
        uint32_t magic = 0;
        input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        ASSERT_TRUE(input.good());
        if (magic != kCheckpointMagic) {
            GTEST_SKIP() << "Current checkpoint is not version1, skip Llama2 version1-only load test";
        }
    }

    const SentencePieceGolden golden = loadSentencePieceGolden();
    ASSERT_FALSE(golden.ids.empty());

    eCEL::Loader loader;
    auto model = loader.load(
        eCEL::ModelType::kLlama2,
        checkpointPath(),
        eUTIL::DeviceType::kCpu);
    auto tokenizer = std::make_unique<eCEL::SentencePieceTokenizer>(tokenizerModelPath());
    eCEL::LLMEngine engine(std::move(model), std::move(tokenizer));

    EXPECT_EQ(engine.tokenizer().name(), "SentencePieceTokenizer");
    EXPECT_TRUE(engine.ropeCache().isBuilt());
    EXPECT_EQ(engine.ropeCache().view().seq_len,
              static_cast<std::size_t>(engine.model().config().seqLen));
    EXPECT_EQ(engine.ropeCache().view().head_size, engine.model().config().headSize);

    const eCEL::GenerationResult result = engine.generate(golden.prompt);
    EXPECT_EQ(result.m_promptIds, golden.ids);
    EXPECT_EQ(result.m_allIds, golden.ids);
    EXPECT_TRUE(result.m_generatedIds.empty());
    EXPECT_TRUE(result.m_generatedText.empty());
    ASSERT_EQ(result.m_modelOutput.size(),
              static_cast<std::size_t>(engine.model().config().vocabSize));
    for (float logit : result.m_modelOutput) {
        EXPECT_TRUE(std::isfinite(logit));
    }
}
