#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "celeritas/tokenizer/SentencePieceTokenizer.h"

namespace {

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

}  // namespace

TEST(test_tokenizer, sentencepiece_encode_decode_round_trip) {
    const std::string modelPath = tokenizerModelPath();
    ASSERT_TRUE(std::filesystem::exists(modelPath));

    eCEL::SentencePieceTokenizer tokenizer(modelPath);
    const SentencePieceGolden golden = loadSentencePieceGolden();

    EXPECT_EQ(tokenizer.name(), "SentencePieceTokenizer");
    EXPECT_GT(tokenizer.vocabSize(), 0u);
    EXPECT_TRUE(tokenizer.specialTokens().unk.has_value());

    const std::vector<eCEL::TokenId> ids = tokenizer.encode(golden.prompt);
    ASSERT_FALSE(ids.empty());
    EXPECT_EQ(ids, golden.ids);

    const std::string decoded = tokenizer.decode(ids);
    EXPECT_EQ(decoded, golden.prompt);
}

TEST(test_tokenizer, sentencepiece_supports_special_tokens_in_encode_and_decode) {
    const std::string modelPath = tokenizerModelPath();
    ASSERT_TRUE(std::filesystem::exists(modelPath));

    eCEL::SentencePieceTokenizer tokenizer(modelPath);
    const std::string text = "Once upon a time";

    const auto& specialTokens = tokenizer.specialTokens();
    ASSERT_TRUE(specialTokens.bos.has_value());
    ASSERT_TRUE(specialTokens.eos.has_value());

    eCEL::EncodeOptions options;
    options.m_addBos = true;
    options.m_addEos = true;

    const std::vector<eCEL::TokenId> ids = tokenizer.encode(text, options);
    ASSERT_GE(ids.size(), 3u);
    EXPECT_EQ(ids.front(), *specialTokens.bos);
    EXPECT_EQ(ids.back(), *specialTokens.eos);
    EXPECT_TRUE(tokenizer.isSpecialToken(ids.front()));
    EXPECT_TRUE(tokenizer.isSpecialToken(ids.back()));

    const std::string skipped = tokenizer.decode(ids);
    EXPECT_EQ(skipped, text);

    eCEL::DecodeOptions decodeOptions;
    decodeOptions.m_skipSpecialTokens = false;
    const std::string withSpecialTokens = tokenizer.decode(ids, decodeOptions);
    EXPECT_FALSE(withSpecialTokens.empty());
}
