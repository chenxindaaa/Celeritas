#pragma once

#include <string>
#include <utility>

#include "udm/common/cudaConfig.h"
#include "udm/core/Tensor.h"

namespace eCEL {

using Tensor = eUTIL::Tensor<float>;

class Workspace {
public:
    virtual ~Workspace() = default;
};

class KVCacheView {
public:
    virtual ~KVCacheView() = default;
};

struct ForwardContext {
    eUTIL::CudaConfig* cuda_config = nullptr;
    int batch_size = 0;
    int q_len = 0;
    int kv_len = 0;
    bool is_prefill = false;
    bool is_decode = false;
    int layer_id = -1;
    Workspace* workspace = nullptr;
    KVCacheView* kv_cache = nullptr;
};

class Layer {
public:
    explicit Layer(eUTIL::DeviceType device) : m_device(device) {}
    virtual ~Layer() = default;

    virtual void forward(const ForwardContext& ctx,
                         const Tensor& input,
                         Tensor& output) = 0;

protected:
    eUTIL::DeviceType m_device;
};

template <typename T>
class Parameter {
public:
    Parameter() = default;

    Parameter(eUTIL::Tensor<T> parameter, std::string name = "")
        : m_parameter(std::move(parameter)),
          m_name(std::move(name)) {}

    const eUTIL::Tensor<T>& view() const { return m_parameter; }
    eUTIL::Tensor<T>& view() { return m_parameter; }

    const std::string& name() const { return m_name; }

private:
    eUTIL::Tensor<T> m_parameter;
    std::string m_name;
};



// class SelfAttentionLayer : public Layer {
// public:
//     SelfAttentionLayer() = default;

//     void forward(const ForwardContext& ctx,
//                  const Tensor& input,
//                  Tensor& output) override;
// };

// class MLPBlockLayer : public Layer {
// public:
//     MLPBlockLayer() = default;

//     void forward(const ForwardContext& ctx,
//                  const Tensor& input,
//                  Tensor& output) override;
// };

// class DecoderLayer : public Layer {
// public:
//     DecoderLayer(RMSNormLayer attn_norm,
//                  SelfAttentionLayer self_attn,
//                  RMSNormLayer ffn_norm,
//                  MLPBlockLayer mlp)
//         : attn_norm_(std::move(attn_norm)),
//           self_attn_(std::move(self_attn)),
//           ffn_norm_(std::move(ffn_norm)),
//           mlp_(std::move(mlp)) {}

//     void forward(const ForwardContext& ctx,
//                  const Tensor& input,
//                  Tensor& output) override;

// private:
//     RMSNormLayer attn_norm_;
//     SelfAttentionLayer self_attn_;
//     RMSNormLayer ffn_norm_;
//     MLPBlockLayer mlp_;
// };



}  // namespace eCEL
