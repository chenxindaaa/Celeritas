#pragma once

// Defines a Llama-style self-attention block built from projections, RoPE, KV cache writes, and MHA.

#include <stdexcept>
#include <type_traits>
#include <utility>

#include "MatmulLayer.h"
#include "MhaLayer.h"

namespace eCEL {

template <typename Tweight>
struct SelfAttentionParams {
    MatmulParams<Tweight> wq;
    MatmulParams<Tweight> wk;
    MatmulParams<Tweight> wv;
    MatmulParams<Tweight> wo;

    SelfAttentionParams() = default;

    SelfAttentionParams(MatmulParams<Tweight> wqParam,
                        MatmulParams<Tweight> wkParam,
                        MatmulParams<Tweight> wvParam,
                        MatmulParams<Tweight> woParam)
        : wq(std::move(wqParam)),
          wk(std::move(wkParam)),
          wv(std::move(wvParam)),
          wo(std::move(woParam)) {}
};

template <typename Tact, typename Tweight = Tact>
class SelfAttentionLayer : public Layer<Tact> {
public:
    static_assert(std::is_same_v<Tact, float>,
                  "SelfAttentionLayer currently supports float activation tensors only");

    using ParamType = SelfAttentionParams<Tweight>;
    using Layer<Tact>::forward;

    explicit SelfAttentionLayer(eUTIL::DeviceType device,
                                ParamType params,
                                int32_t headNum = 0,
                                int32_t seqLen = 0,
                                int32_t kvDim = 0,
                                int32_t kvMul = 0,
                                int32_t headSize = 0)
        : Layer<Tact>(device),
          m_params(std::move(params)),
          m_headNum(headNum),
          m_seqLen(seqLen),
          m_kvDim(kvDim),
          m_kvMul(kvMul),
          m_headSize(headSize),
          m_wqLayer(device, m_params.wq),
          m_wkLayer(device, m_params.wk),
          m_wvLayer(device, m_params.wv),
          m_woLayer(device, m_params.wo),
          m_mhaLayer(device, m_headNum, m_seqLen, m_kvDim, m_kvMul, m_headSize) {}

    explicit SelfAttentionLayer(eUTIL::DeviceType device = eUTIL::DeviceType::kCpu)
        : Layer<Tact>(device),
          m_params(),
          m_headNum(0),
          m_seqLen(0),
          m_kvDim(0),
          m_kvMul(0),
          m_headSize(0),
          m_wqLayer(device),
          m_wkLayer(device),
          m_wvLayer(device),
          m_woLayer(device),
          m_mhaLayer(device) {}

    void forward(const ForwardContext& ctx,
                 TensorListView<Tact> inputs,
                 eUTIL::Tensor<Tact>& output) override;

    void to(eUTIL::DeviceType device) override {
        this->m_device = device;
        m_wqLayer.to(device);
        m_wkLayer.to(device);
        m_wvLayer.to(device);
        m_woLayer.to(device);
        m_mhaLayer.to(device);
        m_params.wq = m_wqLayer.params();
        m_params.wk = m_wkLayer.params();
        m_params.wv = m_wvLayer.params();
        m_params.wo = m_woLayer.params();
    }

    void setParams(ParamType params) {
        m_params = std::move(params);
        m_wqLayer.setParams(m_params.wq);
        m_wkLayer.setParams(m_params.wk);
        m_wvLayer.setParams(m_params.wv);
        m_woLayer.setParams(m_params.wo);
    }

    void setWqParams(MatmulParams<Tweight> params) {
        m_params.wq = std::move(params);
        m_wqLayer.setParams(m_params.wq);
    }

    void setWqWeight(Parameter<Tweight> weight) {
        m_params.wq.weight = std::move(weight);
        m_wqLayer.setWeight(m_params.wq.weight);
    }

    void setWkParams(MatmulParams<Tweight> params) {
        m_params.wk = std::move(params);
        m_wkLayer.setParams(m_params.wk);
    }

    void setWkWeight(Parameter<Tweight> weight) {
        m_params.wk.weight = std::move(weight);
        m_wkLayer.setWeight(m_params.wk.weight);
    }

    void setWvParams(MatmulParams<Tweight> params) {
        m_params.wv = std::move(params);
        m_wvLayer.setParams(m_params.wv);
    }

    void setWvWeight(Parameter<Tweight> weight) {
        m_params.wv.weight = std::move(weight);
        m_wvLayer.setWeight(m_params.wv.weight);
    }

    void setWoParams(MatmulParams<Tweight> params) {
        m_params.wo = std::move(params);
        m_woLayer.setParams(m_params.wo);
    }

    void setWoWeight(Parameter<Tweight> weight) {
        m_params.wo.weight = std::move(weight);
        m_woLayer.setWeight(m_params.wo.weight);
    }

    void setMhaConfig(int32_t headNum,
                      int32_t seqLen,
                      int32_t kvDim,
                      int32_t kvMul,
                      int32_t headSize) {
        m_headNum = headNum;
        m_seqLen = seqLen;
        m_kvDim = kvDim;
        m_kvMul = kvMul;
        m_headSize = headSize;
        m_mhaLayer.setConfig(m_headNum, m_seqLen, m_kvDim, m_kvMul, m_headSize);
    }

    const ParamType& params() const { return m_params; }
    ParamType& params() { return m_params; }
    int32_t headNum() const { return m_headNum; }
    int32_t seqLen() const { return m_seqLen; }
    int32_t kvDim() const { return m_kvDim; }
    int32_t kvMul() const { return m_kvMul; }
    int32_t headSize() const { return m_headSize; }

private:
    // Creates a 1D tensor for one projection output.
    static eUTIL::Tensor<Tact> makeVectorTensor(std::size_t size,
                                                eUTIL::DeviceType device);
    // Resolves the token position for the current attention step.
    int32_t resolveTokenPosition(const ForwardContext& ctx) const;
    // Applies RoPE in place to query and key tensors.
    void applyRope(const ForwardContext& ctx,
                   int32_t tokenPos,
                   eUTIL::Tensor<Tact>& query,
                   eUTIL::Tensor<Tact>& key) const;
    // Stores the current key/value tensors into the active KV cache slot.
    void writeKvCache(const ForwardContext& ctx,
                      int32_t tokenPos,
                      const eUTIL::Tensor<Tact>& key,
                      const eUTIL::Tensor<Tact>& value) const;
    // Validates the runtime input and output shapes for single-token attention.
    void validateTensors(const eUTIL::Tensor<Tact>& input,
                         const eUTIL::Tensor<Tact>& output) const;

private:
    ParamType m_params;
    int32_t m_headNum;
    int32_t m_seqLen;
    int32_t m_kvDim;
    int32_t m_kvMul;
    int32_t m_headSize;
    MatmulLayer<Tact, Tweight> m_wqLayer;
    MatmulLayer<Tact, Tweight> m_wkLayer;
    MatmulLayer<Tact, Tweight> m_wvLayer;
    MatmulLayer<Tact, Tweight> m_woLayer;
    MhaLayer<Tact> m_mhaLayer;
};

}  // namespace eCEL
