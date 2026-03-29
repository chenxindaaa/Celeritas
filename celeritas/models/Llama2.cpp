// Builds Llama2 runtime layers from checkpoint weights and runs the Llama2 forward path.

#include "Llama2.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

#include "celeritas/kvCache/kvCacheMgr.h"
#include "celeritas/ropeCache/RopeCache.h"

namespace eCEL {
namespace {

// Creates one parameter from one checkpoint weight block.
template <typename T = float>
Parameter<T> makeWeightParameter(const RawModelData& rawData,
                                 size_t offset,
                                 std::initializer_list<std::size_t> dims) {
    auto* weightPtr = const_cast<T*>(static_cast<const T*>(rawData.weight(offset)));
    eUTIL::Tensor<T> weight(eUTIL::DeviceType::kCpu, dims, weightPtr, true);
    return Parameter<T>(std::move(weight));
}

// Creates a CPU token-id tensor for embedding lookup.
eUTIL::Tensor<float> makeTokenTensor(const std::vector<TokenId>& inputIds) {
    eUTIL::Tensor<float> tokens(eUTIL::DeviceType::kCpu, inputIds.size());
    for (std::size_t i = 0; i < inputIds.size(); ++i) {
        tokens[static_cast<int>(i)] = static_cast<float>(inputIds[i]);
    }
    return tokens;
}

// Copies one runtime tensor into a std::vector on CPU.
std::vector<float> tensorToVector(eUTIL::Tensor<float> tensor) {
    if (tensor.device() == eUTIL::DeviceType::kCuda) {
        tensor.cpu();
    }

    std::vector<float> values(tensor.size());
    const float* tensorData = tensor.data();
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        values[i] = tensorData[i];
    }
    return values;
}

}  // namespace

// Runs the default forward path with an empty runtime context.
std::vector<float> Llama2::forward(const ModelInputs& inputs) const
{
    ForwardContext ctx;
    return forward(inputs, ctx);
}

// Runs the Llama2 forward path and returns the last-token logits.
std::vector<float> Llama2::forward(const ModelInputs& inputs,
                                   const ForwardContext& ctx) const
{
    if (m_embLayer == nullptr || m_finalNormLayer == nullptr || m_lmHeadLayer == nullptr) {
        throw std::runtime_error("Llama2 layers are not initialized");
    }
    if (m_decoderLayers.size() != static_cast<std::size_t>(m_config.layerNum)) {
        throw std::runtime_error("Llama2 decoder layers are not initialized");
    }
    if (inputs.m_inputIds.empty()) {
        throw std::invalid_argument("Llama2::forward requires non-empty input_ids");
    }
    if (static_cast<int32_t>(inputs.m_inputIds.size()) > m_config.seqLen ||
        static_cast<int32_t>(inputs.m_inputIds.size()) < 0) {
        throw std::invalid_argument("Llama2::forward input_ids exceed model sequence length");
    }

    const std::size_t tokenCount = inputs.m_inputIds.size();
    const std::size_t hiddenSize = static_cast<std::size_t>(m_config.dim);
    const std::size_t vocabSize = static_cast<std::size_t>(m_config.vocabSize);

    std::optional<KVCacheManager> ownedKvCacheManager;
    KVCacheManager* kvCacheManager = ctx.kv_cache_manager;
    if (kvCacheManager == nullptr) {
        ownedKvCacheManager.emplace(m_device, m_config);
        ownedKvCacheManager->allocate();
        kvCacheManager = &ownedKvCacheManager.value();
    }

    std::optional<RopeCache> ownedRopeCache;
    const RopeCacheView* ropeCacheView = ctx.rope_cache;
    if (ropeCacheView == nullptr) {
        ownedRopeCache.emplace(m_device, m_config);
        ownedRopeCache->build();
        ropeCacheView = &ownedRopeCache->view();
    }

    const std::size_t baseKvLen = kvCacheManager->seqLen();
    if (baseKvLen + tokenCount > static_cast<std::size_t>(m_config.seqLen)) {
        throw std::out_of_range("Llama2::forward would exceed kv cache sequence length");
    }

    eUTIL::Tensor<float> input = makeTokenTensor(inputs.m_inputIds);
    if (m_device == eUTIL::DeviceType::kCuda) {
        input.cuda();
    }
    eUTIL::Tensor<float> embeddingOutput(m_device, tokenCount, hiddenSize);
    m_embLayer->forward(ctx, input, embeddingOutput);

    std::vector<float> lastLogits;
    lastLogits.reserve(vocabSize);

    ForwardContext runtimeCtx = ctx;
    runtimeCtx.batch_size = 1;
    runtimeCtx.q_len = 1;
    runtimeCtx.kv_cache_manager = kvCacheManager;
    runtimeCtx.rope_cache = ropeCacheView;
    runtimeCtx.is_prefill = tokenCount > 1;
    runtimeCtx.is_decode = tokenCount == 1 && baseKvLen > 0;

    for (std::size_t tokenIndex = 0; tokenIndex < tokenCount; ++tokenIndex) {
        eUTIL::Tensor<float> hiddenState(
            m_device,
            {hiddenSize},
            embeddingOutput.data() + tokenIndex * hiddenSize,
            true);

        for (std::size_t layerId = 0; layerId < m_decoderLayers.size(); ++layerId) {
            eUTIL::Tensor<float> layerOutput(m_device, hiddenSize);
            ForwardContext layerCtx = runtimeCtx;
            layerCtx.layer_id = static_cast<int>(layerId);
            layerCtx.kv_len = static_cast<int>(baseKvLen + tokenIndex + 1);
            layerCtx.kv_cache = kvCacheManager->getLayerView(static_cast<int32_t>(layerId));
            m_decoderLayers[layerId]->forward(layerCtx, hiddenState, layerOutput);
            hiddenState = std::move(layerOutput);
        }

        eUTIL::Tensor<float> normOutput(m_device, hiddenSize);
        m_finalNormLayer->forward(runtimeCtx, hiddenState, normOutput);

        eUTIL::Tensor<float> logits(m_device, vocabSize);
        m_lmHeadLayer->forward(runtimeCtx, normOutput, logits);
        lastLogits = tensorToVector(std::move(logits));
    }

    return lastLogits;
}

// Validates the loaded config before runtime layers are constructed.
void Llama2::validateConfig() const {
    if (m_config.vocabSize <= 0 || m_config.dim <= 0 || m_config.hiddenDim <= 0 ||
        m_config.layerNum < 0 || m_config.kvDim <= 0 || m_config.seqLen <= 0 ||
        m_config.headNum <= 0 || m_config.kvMul <= 0 || m_config.headSize <= 0) {
        throw std::runtime_error("invalid llama2 config in checkpoint");
    }
}

// Loads weights according to the checkpoint format after all runtime layers exist.
void Llama2::loadWeights() {
    if (m_rawData == nullptr) {
        throw std::runtime_error("raw model data is not initialized");
    }

    if (m_config.checkpointFormat == CheckpointFormat::kVersion1) {
        loadVersion1Weights();
        return;
    }
    loadLegacyWeights();
}

// Loads weights from a version1 checkpoint layout.
void Llama2::loadVersion1Weights() {
    const size_t layerCount = static_cast<size_t>(m_config.layerNum);
    const size_t dim = static_cast<size_t>(m_config.dim);
    const size_t hiddenDim = static_cast<size_t>(m_config.hiddenDim);
    const size_t kvDim = static_cast<size_t>(m_config.kvDim);
    const size_t vocabSize = static_cast<size_t>(m_config.vocabSize);

    size_t offset = 0;

    // Attach all decoder attention norm weights in the same order as version1_export.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setAttnNormWeight(
            makeWeightParameter(*m_rawData, offset, {dim}));
        offset += dim;
    }

    // Attach all decoder FFN norm weights in the same order as version1_export.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setFfnNormWeight(
            makeWeightParameter(*m_rawData, offset, {dim}));
        offset += dim;
    }

    // Attach the final norm weight after all decoder FFN norms.
    m_finalNormLayer->setWeight(makeWeightParameter(*m_rawData, offset, {dim}));
    offset += dim;

    // Attach the embedding weight after the norms, matching version1_export.
    m_embLayer->setWeight(makeWeightParameter(*m_rawData, offset, {vocabSize, dim}));
    offset += vocabSize * dim;

    // Attach all decoder q projection weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setWqWeight(
            makeWeightParameter(*m_rawData, offset, {dim, dim}));
        offset += dim * dim;
    }

    // Attach all decoder k projection weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setWkWeight(
            makeWeightParameter(*m_rawData, offset, {kvDim, dim}));
        offset += kvDim * dim;
    }

    // Attach all decoder v projection weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setWvWeight(
            makeWeightParameter(*m_rawData, offset, {kvDim, dim}));
        offset += kvDim * dim;
    }

    // Attach all decoder o projection weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setWoWeight(
            makeWeightParameter(*m_rawData, offset, {dim, dim}));
        offset += dim * dim;
    }

    // Attach all decoder w1 weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setGateProjWeight(
            makeWeightParameter(*m_rawData, offset, {hiddenDim, dim}));
        offset += hiddenDim * dim;
    }

    // Attach all decoder w2 weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setDownProjWeight(
            makeWeightParameter(*m_rawData, offset, {dim, hiddenDim}));
        offset += dim * hiddenDim;
    }

    // Attach all decoder w3 weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setUpProjWeight(
            makeWeightParameter(*m_rawData, offset, {hiddenDim, dim}));
        offset += hiddenDim * dim;
    }

    if (m_config.isSharedWeight) {
        m_lmHeadLayer->setWeight(m_embLayer->params().weight);
        return;
    }

    m_lmHeadLayer->setWeight(makeWeightParameter(*m_rawData, offset, {vocabSize, dim}));
}

// Loads weights from a legacy checkpoint layout.
void Llama2::loadLegacyWeights() {
    const size_t layerCount = static_cast<size_t>(m_config.layerNum);
    const size_t dim = static_cast<size_t>(m_config.dim);
    const size_t hiddenDim = static_cast<size_t>(m_config.hiddenDim);
    const size_t kvDim = static_cast<size_t>(m_config.kvDim);
    const size_t vocabSize = static_cast<size_t>(m_config.vocabSize);

    size_t offset = 0;

    // Attach legacy embedding weight first, following the original llama2.c layout.
    m_embLayer->setWeight(makeWeightParameter(*m_rawData, offset, {vocabSize, dim}));
    offset += vocabSize * dim;

    // Attach all decoder attention norm weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setAttnNormWeight(
            makeWeightParameter(*m_rawData, offset, {dim}));
        offset += dim;
    }

    // Attach all decoder q projection weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setWqWeight(
            makeWeightParameter(*m_rawData, offset, {dim, dim}));
        offset += dim * dim;
    }

    // Attach all decoder k projection weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setWkWeight(
            makeWeightParameter(*m_rawData, offset, {kvDim, dim}));
        offset += kvDim * dim;
    }

    // Attach all decoder v projection weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setWvWeight(
            makeWeightParameter(*m_rawData, offset, {kvDim, dim}));
        offset += kvDim * dim;
    }

    // Attach all decoder o projection weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setWoWeight(
            makeWeightParameter(*m_rawData, offset, {dim, dim}));
        offset += dim * dim;
    }

    // Attach all decoder FFN norm weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setFfnNormWeight(
            makeWeightParameter(*m_rawData, offset, {dim}));
        offset += dim;
    }

    // Attach all decoder w1 weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setGateProjWeight(
            makeWeightParameter(*m_rawData, offset, {hiddenDim, dim}));
        offset += hiddenDim * dim;
    }

    // Attach all decoder w2 weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setDownProjWeight(
            makeWeightParameter(*m_rawData, offset, {dim, hiddenDim}));
        offset += dim * hiddenDim;
    }

    // Attach all decoder w3 weights.
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers[layerId]->setUpProjWeight(
            makeWeightParameter(*m_rawData, offset, {hiddenDim, dim}));
        offset += hiddenDim * dim;
    }

    // Attach the final norm weight and skip legacy rope tables that are rebuilt at runtime.
    m_finalNormLayer->setWeight(makeWeightParameter(*m_rawData, offset, {dim}));
    offset += dim;
    offset += static_cast<std::size_t>(m_config.seqLen) *
              static_cast<std::size_t>(m_config.headSize);

    if (m_config.isSharedWeight) {
        m_lmHeadLayer->setWeight(m_embLayer->params().weight);
        return;
    }

    m_lmHeadLayer->setWeight(makeWeightParameter(*m_rawData, offset, {vocabSize, dim}));
}

// Creates embedding, decoder, final norm, and lm head runtime layers.
void Llama2::createLayers()
{
    if (m_rawData == nullptr) {
        throw std::runtime_error("raw model data is not initialized");
    }
    validateConfig();

    const size_t layerCount = static_cast<size_t>(m_config.layerNum);

    // Create runtime layers on CPU first and move them later if needed.
    m_embLayer = std::make_unique<EmbLayer<float>>(m_config.vocabSize);
    m_finalNormLayer = std::make_unique<RmsnormLayer<float>>();
    m_lmHeadLayer = std::make_unique<MatmulLayer<float, float>>();

    // Create all decoder layers first, then attach their weights later.
    m_decoderLayers.clear();
    m_decoderLayers.reserve(layerCount);
    for (size_t layerId = 0; layerId < layerCount; ++layerId) {
        m_decoderLayers.push_back(std::make_unique<DecoderLayer<float, float>>());
        m_decoderLayers[layerId]->setMhaConfig(m_config.headNum,
                                               m_config.seqLen,
                                               m_config.kvDim,
                                               m_config.kvMul,
                                               m_config.headSize);
    }
}

void Llama2::moveToDevice(eUTIL::DeviceType device)
{
    if (m_embLayer != nullptr) {
        m_embLayer->to(device);
    }
    if (m_finalNormLayer != nullptr) {
        m_finalNormLayer->to(device);
    }
    if (m_lmHeadLayer != nullptr) {
        m_lmHeadLayer->to(device);
    }
    for (auto& decoderLayer : m_decoderLayers) {
        if (decoderLayer != nullptr) {
            decoderLayer->to(device);
        }
    }
}

}  // namespace eCEL
