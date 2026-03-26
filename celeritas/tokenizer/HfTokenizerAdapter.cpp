#include "HfTokenizerAdapter.h"

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

HfTokenizerAdapter::HfTokenizerAdapter(std::string tokenizerPath)
    : m_tokenizerPath(std::move(tokenizerPath)) {}

std::string HfTokenizerAdapter::name() const {
    return "HfTokenizerAdapter";
}

size_t HfTokenizerAdapter::vocabSize() const {
    return m_vocabSize;
}

std::vector<TokenId> HfTokenizerAdapter::encode(std::string_view text,
                                                const EncodeOptions& options) const {
    (void)text;
    (void)options;
    throw std::logic_error(
        "HfTokenizerAdapter::encode is a skeleton placeholder and is not implemented yet");
}

std::string HfTokenizerAdapter::decode(const std::vector<TokenId>& ids,
                                       const DecodeOptions& options) const {
    (void)ids;
    (void)options;
    throw std::logic_error(
        "HfTokenizerAdapter::decode is a skeleton placeholder and is not implemented yet");
}

bool HfTokenizerAdapter::isSpecialToken(TokenId id) const {
    return matchesSpecialToken(id, m_specialTokens);
}

const SpecialTokens& HfTokenizerAdapter::specialTokens() const {
    return m_specialTokens;
}

}  // namespace eCEL
