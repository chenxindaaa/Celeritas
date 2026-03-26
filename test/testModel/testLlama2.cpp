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

#include "celeritas/engine/LLMEgine.h"
#include "celeritas/models/Llama2.h"
#include "celeritas/models/config.h"
#include "celeritas/tokenizer/SentencePieceTokenizer.h"

namespace {

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

    eCEL::ModelConfig config{};
    input.read(reinterpret_cast<char*>(&config), sizeof(config));
    if (!input) {
        throw std::runtime_error("Failed to read ModelConfig from checkpoint: " + modelPath);
    }
    if (config.dim <= 0 || config.vocab_size == 0) {
        throw std::runtime_error("Invalid ModelConfig in checkpoint: " + modelPath);
    }

    const int32_t vocabSize = std::abs(config.vocab_size);
    if (tokenId < 0 || tokenId >= vocabSize) {
        throw std::out_of_range("Token id is out of checkpoint vocab range");
    }

    const std::streamoff rowOffset =
        static_cast<std::streamoff>(sizeof(eCEL::ModelConfig)) +
        static_cast<std::streamoff>(tokenId) * config.dim * static_cast<std::streamoff>(sizeof(float));
    input.seekg(rowOffset);
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
     llm_engine_with_sentencepiece_tokenizer_forwards_first_token_embedding_from_checkpoint) {
    ASSERT_TRUE(std::filesystem::exists(checkpointPath()));
    ASSERT_TRUE(std::filesystem::exists(tokenizerModelPath()));
    ASSERT_TRUE(std::filesystem::exists(tokenizerGoldenPath()));

    const SentencePieceGolden golden = loadSentencePieceGolden();
    ASSERT_FALSE(golden.ids.empty());

    auto model = std::make_unique<eCEL::Llama2>(
        checkpointPath(),
        tokenizerModelPath(),
        eUTIL::DeviceType::kCpu);
    auto tokenizer = std::make_unique<eCEL::SentencePieceTokenizer>(tokenizerModelPath());
    eCEL::LLMEngine engine(std::move(model), std::move(tokenizer));

    EXPECT_EQ(engine.tokenizer().name(), "SentencePieceTokenizer");

    const eCEL::GenerationResult result = engine.generate(golden.prompt);
    EXPECT_EQ(result.m_promptIds, golden.ids);
    EXPECT_EQ(result.m_allIds, golden.ids);

    int32_t hiddenSize = 0;
    const std::vector<float> expectedFirstTokenEmbedding =
        loadExpectedEmbeddingRow(golden.ids.front(), &hiddenSize);
    ASSERT_GT(hiddenSize, 0);
    EXPECT_EQ(engine.model().config().dim_, hiddenSize);
    ASSERT_EQ(expectedFirstTokenEmbedding.size(), static_cast<std::size_t>(hiddenSize));
    ASSERT_EQ(result.m_modelOutput.size(),
              golden.ids.size() * static_cast<std::size_t>(hiddenSize));

    for (int32_t i = 0; i < hiddenSize; ++i) {
        EXPECT_NEAR(result.m_modelOutput[static_cast<std::size_t>(i)],
                    expectedFirstTokenEmbedding[static_cast<std::size_t>(i)],
                    1e-5f)
            << "Mismatch at hidden index " << i;
    }
}
