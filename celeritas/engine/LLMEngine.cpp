#include "LLMEngine.h"

#include <stdexcept>
#include <utility>

#include "celeritas/sampler/argmax_sampler.h"

namespace eCEL {

LLMEngine::LLMEngine(std::unique_ptr<Model> model,
                     std::unique_ptr<Tokenizer> tokenizer,
                     std::unique_ptr<Sampler> sampler)
    : m_model(std::move(model)),
      m_tokenizer(std::move(tokenizer)),
      m_sampler(std::move(sampler)) {
    if (m_model == nullptr) {
        throw std::invalid_argument("LLMEngine requires a non-null model");
    }
    if (m_tokenizer == nullptr) {
        throw std::invalid_argument("LLMEngine requires a non-null tokenizer");
    }
    if (!m_model->isLoaded()) {
        throw std::invalid_argument("LLMEngine requires a loaded model");
    }
    if (m_sampler == nullptr) {
        m_sampler = std::make_unique<ArgmaxSampler>(m_model->device());
    }
    m_kvCacheManager = std::make_unique<KVCacheManager>(m_model->device(), m_model->config());
    m_kvCacheManager->allocate();
    m_ropeCache = std::make_unique<RopeCache>(m_model->device(), m_model->config());
    m_ropeCache->build();
}

GenerationResult LLMEngine::generate(std::string_view prompt,
                                     const GenerationOptions& options) {
    GenerationResult result;
    result.m_promptIds = encodePrompt(prompt, options.m_encodeOptions);
    if (result.m_promptIds.empty()) {
        throw std::invalid_argument("LLMEngine prompt encoded to an empty token sequence");
    }
    result.m_allIds = result.m_promptIds;

    const ModelInputs promptInputs = buildModelInputs(result.m_promptIds);
    result.m_modelOutput = forwardModel(promptInputs);
    m_kvCacheManager->appendTokens(result.m_promptIds.size());

    const std::optional<TokenId> eosToken = m_tokenizer->specialTokens().eos;
    for (std::size_t step = 0; step < options.m_maxNewTokens; ++step) {
        const TokenId nextId = static_cast<TokenId>(m_sampler->sample(result.m_modelOutput));
        result.m_generatedIds.push_back(nextId);
        result.m_allIds.push_back(nextId);

        if (options.m_stopAtEos && eosToken.has_value() && nextId == *eosToken) {
            break;
        }

        const ModelInputs nextInputs =
            ModelInputBuilder::singleToken(nextId,
                                           static_cast<int32_t>(result.m_allIds.size() - 1));
        result.m_modelOutput = forwardModel(nextInputs);
        m_kvCacheManager->appendTokens(1);
    }

    result.m_generatedText =
        m_tokenizer->decode(result.m_generatedIds, options.m_decodeOptions);

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

std::vector<float> LLMEngine::forwardModel(const ModelInputs& inputs) {
    ForwardContext ctx;
    ctx.rope_cache = &m_ropeCache->view();
    ctx.kv_cache_manager = m_kvCacheManager.get();
    return m_model->forward(inputs, ctx);
}

}  // namespace eCEL
