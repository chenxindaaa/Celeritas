#pragma once

#include <string>
#include <vector>

#include "celeritas/tokenizer/Tokenizer.h"

namespace eCEL {

class BpeTokenizer : public Tokenizer {
public:
    explicit BpeTokenizer(std::string modelPath);

    std::string name() const override;
    size_t vocabSize() const override;

    std::vector<TokenId> encode(
        std::string_view text,
        const EncodeOptions& options = {}) const override;

    std::string decode(
        const std::vector<TokenId>& ids,
        const DecodeOptions& options = {}) const override;

    bool isSpecialToken(TokenId id) const override;
    const SpecialTokens& specialTokens() const override;

    const std::string& modelPath() const { return m_modelPath; }

private:
    std::string m_modelPath;
    size_t m_vocabSize = 0;
    SpecialTokens m_specialTokens;
};

}  // namespace eCEL
