#pragma once

// Defines the multi-head attention layer that consumes query input and KV cache view.

#include <cstdint>
#include <type_traits>

#include "Layer.h"
#include "celeritas/kernels/KernelFactory.h"

namespace eCEL {

template <typename T>
class MhaLayer : public Layer<T> {
public:
    static_assert(std::is_same_v<T, float>,
                  "MhaLayer currently supports float tensors only");

    using Layer<T>::forward;

    explicit MhaLayer(eUTIL::DeviceType device,
                      int32_t headNum,
                      int32_t seqLen,
                      int32_t kvDim,
                      int32_t kvMul,
                      int32_t headSize)
        : Layer<T>(device),
          m_headNum(headNum),
          m_seqLen(seqLen),
          m_kvDim(kvDim),
          m_kvMul(kvMul),
          m_headSize(headSize) {}

    explicit MhaLayer(eUTIL::DeviceType device = eUTIL::DeviceType::kCpu)
        : Layer<T>(device),
          m_headNum(0),
          m_seqLen(0),
          m_kvDim(0),
          m_kvMul(0),
          m_headSize(0) {}

    void forward(const ForwardContext& ctx,
                 TensorListView<T> inputs,
                 eUTIL::Tensor<T>& output) override;

    void to(eUTIL::DeviceType device) override {
        this->m_device = device;
    }

    void setConfig(int32_t headNum,
                   int32_t seqLen,
                   int32_t kvDim,
                   int32_t kvMul,
                   int32_t headSize) {
        m_headNum = headNum;
        m_seqLen = seqLen;
        m_kvDim = kvDim;
        m_kvMul = kvMul;
        m_headSize = headSize;
    }

    int32_t headNum() const { return m_headNum; }
    int32_t seqLen() const { return m_seqLen; }
    int32_t kvDim() const { return m_kvDim; }
    int32_t kvMul() const { return m_kvMul; }
    int32_t headSize() const { return m_headSize; }

private:
    // Validates tensor shapes against the configured attention dimensions.
    void validateTensorShapes(const eUTIL::Tensor<T>& query,
                              const eUTIL::Tensor<T>& output) const;

private:
    int32_t m_headNum;
    int32_t m_seqLen;
    int32_t m_kvDim;
    int32_t m_kvMul;
    int32_t m_headSize;
};

}  // namespace eCEL
