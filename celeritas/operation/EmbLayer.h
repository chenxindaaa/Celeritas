#pragma once

// Defines the embedding layer that maps token ids to embedding vectors.

#include <cstdint>
#include <type_traits>
#include <utility>

#include "Layer.h"
#include "celeritas/kernels/KernelFactory.h"

namespace eCEL {

template <typename T>
struct EmbParams {
    Parameter<T> weight;

    EmbParams() = default;

    explicit EmbParams(Parameter<T> weight_param)
        : weight(std::move(weight_param)) {}
};

template <typename T>
class EmbLayer : public Layer<T> {
public:
    using ParamType = EmbParams<T>;
    using Layer<T>::forward;

    explicit EmbLayer(int32_t vocab_size,
                      eUTIL::DeviceType device = eUTIL::DeviceType::kCpu)
        : Layer<T>(device),
          m_vocabSize(vocab_size),
          m_params() {}

    explicit EmbLayer(eUTIL::DeviceType device,
                      ParamType params,
                      int32_t vocab_size)
        : Layer<T>(device),
          m_vocabSize(vocab_size),
          m_params(std::move(params)) {}

    explicit EmbLayer(eUTIL::DeviceType device,
                      Parameter<T> weight,
                      int32_t vocab_size)
        : Layer<T>(device),
          m_vocabSize(vocab_size),
          m_params(std::move(weight)) {}

    void forward(const ForwardContext& ctx,
                 TensorListView<T> inputs,
                 eUTIL::Tensor<T>& output) override;

    void to(eUTIL::DeviceType device) override {
        this->m_device = device;
        m_params.weight.to(device);
    }

    void setParams(ParamType params) { m_params = std::move(params); }
    void setWeight(Parameter<T> weight) { m_params.weight = std::move(weight); }
    void setVocabSize(int32_t vocab_size) { m_vocabSize = vocab_size; }
    int32_t vocabSize() const { return m_vocabSize; }
    const ParamType& params() const { return m_params; }
    ParamType& params() { return m_params; }

private:
    int32_t m_vocabSize;
    ParamType m_params;
};

}  // namespace eCEL
