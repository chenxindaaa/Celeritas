#include "SentencePieceTokenizer.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace eCEL {
namespace {

std::optional<TokenId> makeOptionalTokenId(int id) {
    if (id < 0) {
        return std::nullopt;
    }
    return static_cast<TokenId>(id);
}

bool matchesSpecialToken(TokenId id, const SpecialTokens& tokens) {
    if (tokens.bos && id == *tokens.bos) {
        return true;
    }
    if (tokens.eos && id == *tokens.eos) {
        return true;
    }
    if (tokens.pad && id == *tokens.pad) {
        return true;
    }
    if (tokens.unk && id == *tokens.unk) {
        return true;
    }
    for (const auto& [name, tokenId] : tokens.extra) {
        (void)name;
        if (id == tokenId) {
            return true;
        }
    }
    return false;
}

}  // namespace

SentencePieceTokenizer::SentencePieceTokenizer(std::string modelPath)
    : m_modelPath(std::move(modelPath)),
      m_processor(std::make_unique<sentencepiece::SentencePieceProcessor>()) {
    if (m_processor == nullptr) {
        throw std::runtime_error("Failed to create SentencePieceProcessor");
    }

    const auto status = m_processor->Load(m_modelPath);
    if (!status.ok()) {
        throw std::runtime_error(
            "Failed to load sentencepiece model from: " + m_modelPath +
            ", error: " + status.ToString());
    }

    m_vocabSize = static_cast<size_t>(m_processor->GetPieceSize());
    m_specialTokens.bos = makeOptionalTokenId(m_processor->bos_id());
    m_specialTokens.eos = makeOptionalTokenId(m_processor->eos_id());
    m_specialTokens.pad = makeOptionalTokenId(m_processor->pad_id());
    m_specialTokens.unk = makeOptionalTokenId(m_processor->unk_id());
}

std::string SentencePieceTokenizer::name() const {
    return "SentencePieceTokenizer";
}

size_t SentencePieceTokenizer::vocabSize() const {
    return m_vocabSize;
}

std::vector<TokenId> SentencePieceTokenizer::encode(
    std::string_view text,
    const EncodeOptions& options) const {
    if (m_processor == nullptr) {
        throw std::runtime_error("SentencePieceTokenizer is not initialized");
    }

    std::vector<int> encodedIds = m_processor->EncodeAsIds(text);
    std::vector<TokenId> tokenIds;
    tokenIds.reserve(
        encodedIds.size() + (options.m_addBos ? 1u : 0u) + (options.m_addEos ? 1u : 0u));

    if (options.m_addBos && m_specialTokens.bos.has_value()) {
        tokenIds.push_back(*m_specialTokens.bos);
    }

    std::transform(encodedIds.begin(),
                   encodedIds.end(),
                   std::back_inserter(tokenIds),
                   [](int id) { return static_cast<TokenId>(id); });

    if (options.m_addEos && m_specialTokens.eos.has_value()) {
        tokenIds.push_back(*m_specialTokens.eos);
    }

    return tokenIds;
}

std::string SentencePieceTokenizer::decode(const std::vector<TokenId>& ids,
                                           const DecodeOptions& options) const {
    if (m_processor == nullptr) {
        throw std::runtime_error("SentencePieceTokenizer is not initialized");
    }

    std::vector<int> decodeIds;
    decodeIds.reserve(ids.size());
    for (const TokenId id : ids) {
        if (options.m_skipSpecialTokens && isSpecialToken(id)) {
            continue;
        }
        decodeIds.push_back(static_cast<int>(id));
    }

    if (decodeIds.empty()) {
        return "";
    }

    return m_processor->DecodeIds(decodeIds);
}

bool SentencePieceTokenizer::isSpecialToken(TokenId id) const {
    return matchesSpecialToken(id, m_specialTokens);
}

const SpecialTokens& SentencePieceTokenizer::specialTokens() const {
    return m_specialTokens;
}

}  // namespace eCEL
