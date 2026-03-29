#include "SelfAttentionLayer.h"

#include <cstddef>
#include <stdexcept>

#include "celeritas/kvCache/kvCacheMgr.h"
#include "celeritas/ropeCache/RopeCache.h"
#include "udm/memory/MemoryMgr.h"

namespace eCEL {

template <typename Tact, typename Tweight>
eUTIL::Tensor<Tact> SelfAttentionLayer<Tact, Tweight>::makeVectorTensor(std::size_t size,
                                                                        eUTIL::DeviceType device) {
    return eUTIL::Tensor<Tact>(device, size);
}

template <typename Tact, typename Tweight>
int32_t SelfAttentionLayer<Tact, Tweight>::resolveTokenPosition(const ForwardContext& ctx) const {
    if (ctx.kv_cache == nullptr) {
        throw std::invalid_argument("SelfAttentionLayer requires kv_cache in ForwardContext");
    }

    const std::size_t cachedSeqLen = ctx.kv_cache->seq_len;
    if (ctx.kv_len > 0) {
        const std::size_t requestedSeqLen = static_cast<std::size_t>(ctx.kv_len);
        if (requestedSeqLen > cachedSeqLen) {
            return static_cast<int32_t>(requestedSeqLen - 1);
        }
        return static_cast<int32_t>(requestedSeqLen);
    }

    return static_cast<int32_t>(cachedSeqLen);
}

template <typename Tact, typename Tweight>
void SelfAttentionLayer<Tact, Tweight>::applyRope(const ForwardContext& ctx,
                                                  int32_t tokenPos,
                                                  eUTIL::Tensor<Tact>& query,
                                                  eUTIL::Tensor<Tact>& key) const {
    if (ctx.rope_cache == nullptr) {
        throw std::invalid_argument("SelfAttentionLayer requires rope_cache in ForwardContext");
    }
    if (ctx.rope_cache->sin_ptr == nullptr || ctx.rope_cache->cos_ptr == nullptr) {
        throw std::invalid_argument("SelfAttentionLayer requires valid rope cache pointers");
    }
    if (ctx.rope_cache->head_size != m_headSize) {
        throw std::invalid_argument("SelfAttentionLayer rope cache head size does not match MHA config");
    }
    if (tokenPos < 0 || static_cast<std::size_t>(tokenPos) >= ctx.rope_cache->seq_len) {
        throw std::out_of_range("SelfAttentionLayer token position exceeds rope cache length");
    }

    eUTIL::Tensor<Tact> sinCache(
        this->m_device,
        {ctx.rope_cache->seq_len, static_cast<std::size_t>(ctx.rope_cache->head_size)},
        const_cast<Tact*>(reinterpret_cast<const Tact*>(ctx.rope_cache->sin_ptr)),
        true);
    eUTIL::Tensor<Tact> cosCache(
        this->m_device,
        {ctx.rope_cache->seq_len, static_cast<std::size_t>(ctx.rope_cache->head_size)},
        const_cast<Tact*>(reinterpret_cast<const Tact*>(ctx.rope_cache->cos_ptr)),
        true);

    auto kernel = KernelFactory::getRopeKernel();
    kernel(tokenPos,
           m_headNum * m_headSize,
           m_kvDim,
           m_headSize,
           query,
           key,
           sinCache,
           cosCache,
           ctx.cuda_config ? ctx.cuda_config->stream : nullptr);
}

template <typename Tact, typename Tweight>
void SelfAttentionLayer<Tact, Tweight>::writeKvCache(const ForwardContext& ctx,
                                                     int32_t tokenPos,
                                                     const eUTIL::Tensor<Tact>& key,
                                                     const eUTIL::Tensor<Tact>& value) const {
    if (ctx.kv_cache == nullptr) {
        throw std::invalid_argument("SelfAttentionLayer requires kv_cache in ForwardContext");
    }
    if (ctx.kv_cache->k_ptr == nullptr || ctx.kv_cache->v_ptr == nullptr) {
        throw std::invalid_argument("SelfAttentionLayer requires valid KV cache pointers");
    }
    if (tokenPos < 0 || tokenPos >= m_seqLen) {
        throw std::out_of_range("SelfAttentionLayer token position exceeds configured seq_len");
    }

    const std::size_t kvDim = static_cast<std::size_t>(m_kvDim);
    const std::size_t cacheOffset = static_cast<std::size_t>(tokenPos) * kvDim;
    auto& memoryMgr = eUTIL::MemoryMgr::getInstance();
    const cudaStream_t stream = ctx.cuda_config ? ctx.cuda_config->stream : nullptr;

    memoryMgr.memcpy(this->m_device,
                     ctx.kv_cache->k_ptr + cacheOffset,
                     this->m_device,
                     key.data(),
                     kvDim * sizeof(Tact),
                     stream);
    memoryMgr.memcpy(this->m_device,
                     ctx.kv_cache->v_ptr + cacheOffset,
                     this->m_device,
                     value.data(),
                     kvDim * sizeof(Tact),
                     stream);
}

template <typename Tact, typename Tweight>
void SelfAttentionLayer<Tact, Tweight>::validateTensors(const eUTIL::Tensor<Tact>& input,
                                                        const eUTIL::Tensor<Tact>& output) const {
    if (input.dimSize() != 1) {
        throw std::invalid_argument("SelfAttentionLayer currently supports 1D single-token input only");
    }

    const std::size_t hiddenSize =
        static_cast<std::size_t>(m_headNum) * static_cast<std::size_t>(m_headSize);
    if (input.size() != hiddenSize) {
        throw std::invalid_argument("SelfAttentionLayer input size does not match configured hidden size");
    }
    if (output.size() != hiddenSize) {
        throw std::invalid_argument("SelfAttentionLayer output size does not match configured hidden size");
    }
}

template <typename Tact, typename Tweight>
void SelfAttentionLayer<Tact, Tweight>::forward(const ForwardContext& ctx,
                                                TensorListView<Tact> inputs,
                                                eUTIL::Tensor<Tact>& output) {
    Layer<Tact>::requireInputCount(inputs, 1, "SelfAttentionLayer");

    const eUTIL::Tensor<Tact>& input = inputs[0];
    validateTensors(input, output);

    const std::size_t hiddenSize =
        static_cast<std::size_t>(m_headNum) * static_cast<std::size_t>(m_headSize);
    const std::size_t kvSize = static_cast<std::size_t>(m_kvDim);
    eUTIL::Tensor<Tact> query = makeVectorTensor(hiddenSize, this->m_device);
    eUTIL::Tensor<Tact> key = makeVectorTensor(kvSize, this->m_device);
    eUTIL::Tensor<Tact> value = makeVectorTensor(kvSize, this->m_device);

    m_wqLayer.forward(ctx, input, query);
    m_wkLayer.forward(ctx, input, key);
    m_wvLayer.forward(ctx, input, value);

    const int32_t tokenPos = resolveTokenPosition(ctx);
    applyRope(ctx, tokenPos, query, key);
    writeKvCache(ctx, tokenPos, key, value);

    ForwardContext mhaCtx = ctx;
    mhaCtx.kv_len = tokenPos + 1;
    eUTIL::Tensor<Tact> attnOutput = makeVectorTensor(hiddenSize, this->m_device);
    m_mhaLayer.forward(mhaCtx, query, attnOutput);

    m_woLayer.forward(ctx, attnOutput, output);
}

template class SelfAttentionLayer<float, float>;

}  // namespace eCEL
