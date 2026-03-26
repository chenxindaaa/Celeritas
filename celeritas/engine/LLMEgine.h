#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "celeritas/models/Model.h"
#include "celeritas/tokenizer/Tokenizer.h"

namespace eCEL {

struct GenerationOptions {
    EncodeOptions m_encodeOptions;
    DecodeOptions m_decodeOptions;
    std::size_t m_maxNewTokens = 1;
    bool m_stopAtEos = true;
};

struct GenerationResult {
    std::vector<TokenId> m_promptIds;
    std::vector<TokenId> m_generatedIds;
    std::vector<TokenId> m_allIds;
    std::string m_generatedText;
    std::vector<float> m_modelOutput;
};

class LLMEngine {
public:
    LLMEngine(std::unique_ptr<Model> model,
              std::unique_ptr<Tokenizer> tokenizer);

    GenerationResult generate(std::string_view prompt,
                              const GenerationOptions& options = {}) const;

    const Model& model() const { return *m_model; }
    const Tokenizer& tokenizer() const { return *m_tokenizer; }

private:
    LLMEngine() noexcept = default;
    std::vector<TokenId> encodePrompt(std::string_view prompt,
                                       const EncodeOptions& options = {}) const;
    ModelInputs buildModelInputs(const std::vector<TokenId>& ids) const;
    std::vector<float> forwardModel(const ModelInputs& inputs) const;

private:
    std::unique_ptr<Model> m_model;
    std::unique_ptr<Tokenizer> m_tokenizer;
};

}  // namespace eCEL
