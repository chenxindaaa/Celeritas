#include "BpeTokenizer.h"

#include <stdexcept>
#include <utility>

namespace eCEL {
namespace {

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

BpeTokenizer::BpeTokenizer(std::string modelPath)
    : m_modelPath(std::move(modelPath)) {}

std::string BpeTokenizer::name() const {
    return "BpeTokenizer";
}

size_t BpeTokenizer::vocabSize() const {
    return m_vocabSize;
}

std::vector<TokenId> BpeTokenizer::encode(std::string_view text,
                                          const EncodeOptions& options) const {
    (void)text;
    (void)options;
    throw std::logic_error(
        "BpeTokenizer::encode is a skeleton placeholder and is not implemented yet");
}

std::string BpeTokenizer::decode(const std::vector<TokenId>& ids,
                                 const DecodeOptions& options) const {
    (void)ids;
    (void)options;
    throw std::logic_error(
        "BpeTokenizer::decode is a skeleton placeholder and is not implemented yet");
}

bool BpeTokenizer::isSpecialToken(TokenId id) const {
    return matchesSpecialToken(id, m_specialTokens);
}

const SpecialTokens& BpeTokenizer::specialTokens() const {
    return m_specialTokens;
}

}  // namespace eCEL
