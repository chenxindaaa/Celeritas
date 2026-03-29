#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "celeritas/kvCache/kvCacheMgr.h"
#include "celeritas/models/Model.h"
#include "celeritas/ropeCache/RopeCache.h"
#include "celeritas/sampler/sampler.h"
#include "celeritas/tokenizer/Tokenizer.h"

namespace eCEL {

struct GenerationOptions {
    EncodeOptions m_encodeOptions;
    DecodeOptions m_decodeOptions;
    std::size_t m_maxNewTokens = 0;
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
              std::unique_ptr<Tokenizer> tokenizer,
              std::unique_ptr<Sampler> sampler = nullptr);

    GenerationResult generate(std::string_view prompt,
                              const GenerationOptions& options = {});

    const Model& model() const { return *m_model; }
    const Tokenizer& tokenizer() const { return *m_tokenizer; }
    const Sampler& sampler() const { return *m_sampler; }
    const KVCacheManager& kvCacheManager() const { return *m_kvCacheManager; }
    const RopeCache& ropeCache() const { return *m_ropeCache; }

private:
    LLMEngine() noexcept = default;
    std::vector<TokenId> encodePrompt(std::string_view prompt,
                                       const EncodeOptions& options = {}) const;
    ModelInputs buildModelInputs(const std::vector<TokenId>& ids) const;
    std::vector<float> forwardModel(const ModelInputs& inputs);

private:
    std::unique_ptr<Model> m_model;
    std::unique_ptr<Tokenizer> m_tokenizer;
    std::unique_ptr<Sampler> m_sampler;
    std::unique_ptr<KVCacheManager> m_kvCacheManager;
    std::unique_ptr<RopeCache> m_ropeCache;
};

}  // namespace eCEL
