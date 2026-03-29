#pragma once

// Defines a Llama-style decoder layer composed of RMSNorm, self-attention, MLP, and residual adds.

#include <type_traits>
#include <utility>

#include "AddLayer.h"
#include "MlpLayer.h"
#include "RmsnormLayer.h"
#include "SelfAttentionLayer.h"

namespace eCEL {

template <typename Tweight>
struct DecoderParams {
    RmsnormParams<float> attn_norm;
    SelfAttentionParams<Tweight> self_attention;
    RmsnormParams<float> ffn_norm;
    MlpParams<Tweight> mlp;

    DecoderParams() = default;

    DecoderParams(RmsnormParams<float> attnNormParam,
                  SelfAttentionParams<Tweight> selfAttentionParam,
                  RmsnormParams<float> ffnNormParam,
                  MlpParams<Tweight> mlpParam)
        : attn_norm(std::move(attnNormParam)),
          self_attention(std::move(selfAttentionParam)),
          ffn_norm(std::move(ffnNormParam)),
          mlp(std::move(mlpParam)) {}
};

template <typename Tact, typename Tweight = Tact>
class DecoderLayer : public Layer<Tact> {
public:
    static_assert(std::is_same_v<Tact, float>,
                  "DecoderLayer currently supports float activation tensors only");

    using ParamType = DecoderParams<Tweight>;
    using Layer<Tact>::forward;

    explicit DecoderLayer(eUTIL::DeviceType device,
                          ParamType params,
                          int32_t headNum = 0,
                          int32_t seqLen = 0,
                          int32_t kvDim = 0,
                          int32_t kvMul = 0,
                          int32_t headSize = 0)
        : Layer<Tact>(device),
          m_attnNormLayer(device, std::move(params.attn_norm)),
          m_selfAttentionLayer(device,
                               std::move(params.self_attention),
                               headNum,
                               seqLen,
                               kvDim,
                               kvMul,
                               headSize),
          m_ffnNormLayer(device, std::move(params.ffn_norm)),
          m_mlpLayer(device, std::move(params.mlp)),
          m_addLayer(device) {}

    explicit DecoderLayer(eUTIL::DeviceType device = eUTIL::DeviceType::kCpu)
        : Layer<Tact>(device),
          m_attnNormLayer(device),
          m_selfAttentionLayer(device),
          m_ffnNormLayer(device),
          m_mlpLayer(device),
          m_addLayer(device) {}

    void forward(const ForwardContext& ctx,
                 TensorListView<Tact> inputs,
                 eUTIL::Tensor<Tact>& output) override;

    void to(eUTIL::DeviceType device) override {
        this->m_device = device;
        m_attnNormLayer.to(device);
        m_selfAttentionLayer.to(device);
        m_ffnNormLayer.to(device);
        m_mlpLayer.to(device);
        m_addLayer.to(device);
    }

    void setParams(ParamType params) {
        m_attnNormLayer.setParams(std::move(params.attn_norm));
        m_selfAttentionLayer.setParams(std::move(params.self_attention));
        m_ffnNormLayer.setParams(std::move(params.ffn_norm));
        m_mlpLayer.setParams(std::move(params.mlp));
    }

    void setAttnNormParams(RmsnormParams<float> params) {
        m_attnNormLayer.setParams(std::move(params));
    }

    void setAttnNormWeight(Parameter<float> weight) {
        m_attnNormLayer.setWeight(std::move(weight));
    }

    void setSelfAttentionParams(SelfAttentionParams<Tweight> params) {
        m_selfAttentionLayer.setParams(std::move(params));
    }

    void setFfnNormParams(RmsnormParams<float> params) {
        m_ffnNormLayer.setParams(std::move(params));
    }

    void setFfnNormWeight(Parameter<float> weight) {
        m_ffnNormLayer.setWeight(std::move(weight));
    }

    void setMlpParams(MlpParams<Tweight> params) {
        m_mlpLayer.setParams(std::move(params));
    }

    void setWqParams(MatmulParams<Tweight> params) {
        m_selfAttentionLayer.setWqParams(std::move(params));
    }

    void setWqWeight(Parameter<Tweight> weight) {
        m_selfAttentionLayer.setWqWeight(std::move(weight));
    }

    void setWkParams(MatmulParams<Tweight> params) {
        m_selfAttentionLayer.setWkParams(std::move(params));
    }

    void setWkWeight(Parameter<Tweight> weight) {
        m_selfAttentionLayer.setWkWeight(std::move(weight));
    }

    void setWvParams(MatmulParams<Tweight> params) {
        m_selfAttentionLayer.setWvParams(std::move(params));
    }

    void setWvWeight(Parameter<Tweight> weight) {
        m_selfAttentionLayer.setWvWeight(std::move(weight));
    }

    void setWoParams(MatmulParams<Tweight> params) {
        m_selfAttentionLayer.setWoParams(std::move(params));
    }

    void setWoWeight(Parameter<Tweight> weight) {
        m_selfAttentionLayer.setWoWeight(std::move(weight));
    }

    void setMhaConfig(int32_t headNum,
                      int32_t seqLen,
                      int32_t kvDim,
                      int32_t kvMul,
                      int32_t headSize) {
        m_selfAttentionLayer.setMhaConfig(headNum, seqLen, kvDim, kvMul, headSize);
    }

    void setGateProjParams(MatmulParams<Tweight> params) {
        m_mlpLayer.setGateProjParams(std::move(params));
    }

    void setGateProjWeight(Parameter<Tweight> weight) {
        m_mlpLayer.setGateProjWeight(std::move(weight));
    }

    void setDownProjParams(MatmulParams<Tweight> params) {
        m_mlpLayer.setDownProjParams(std::move(params));
    }

    void setDownProjWeight(Parameter<Tweight> weight) {
        m_mlpLayer.setDownProjWeight(std::move(weight));
    }

    void setUpProjParams(MatmulParams<Tweight> params) {
        m_mlpLayer.setUpProjParams(std::move(params));
    }

    void setUpProjWeight(Parameter<Tweight> weight) {
        m_mlpLayer.setUpProjWeight(std::move(weight));
    }

    ParamType params() const {
        return ParamType(
            m_attnNormLayer.params(),
            m_selfAttentionLayer.params(),
            m_ffnNormLayer.params(),
            m_mlpLayer.params());
    }

    int32_t headNum() const { return m_selfAttentionLayer.headNum(); }
    int32_t seqLen() const { return m_selfAttentionLayer.seqLen(); }
    int32_t kvDim() const { return m_selfAttentionLayer.kvDim(); }
    int32_t kvMul() const { return m_selfAttentionLayer.kvMul(); }
    int32_t headSize() const { return m_selfAttentionLayer.headSize(); }

private:
    RmsnormLayer<Tact> m_attnNormLayer;
    SelfAttentionLayer<Tact, Tweight> m_selfAttentionLayer;
    RmsnormLayer<Tact> m_ffnNormLayer;
    MlpLayer<Tact, Tweight> m_mlpLayer;
    AddLayer<Tact> m_addLayer;
};

}  // namespace eCEL
