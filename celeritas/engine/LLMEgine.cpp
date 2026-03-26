#include "LLMEgine.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace eCEL {

LLMEngine::LLMEngine(std::unique_ptr<Model> model,
                     std::unique_ptr<Tokenizer> tokenizer)
    : m_model(std::move(model)),
      m_tokenizer(std::move(tokenizer)) {
    m_model->init();
}

GenerationResult LLMEngine::generate(std::string_view prompt,
                                     const GenerationOptions& options) const {
    GenerationResult result;
    result.m_promptIds = encodePrompt(prompt, options.m_encodeOptions);
    result.m_allIds = result.m_promptIds;
    const ModelInputs inputs = buildModelInputs(result.m_allIds);
    result.m_modelOutput = forwardModel(inputs);

    return result;
}

std::vector<TokenId> LLMEngine::encodePrompt(
    std::string_view prompt,
    const EncodeOptions& options) const {
    return m_tokenizer->encode(prompt, options);
}

ModelInputs LLMEngine::buildModelInputs(const std::vector<TokenId>& ids) const {
    return ModelInputBuilder::fromIds(ids, m_tokenizer->specialTokens().pad);
}

std::vector<float> LLMEngine::forwardModel(const ModelInputs& inputs) const {
    return m_model->forward(inputs);
}

}  // namespace eCEL
