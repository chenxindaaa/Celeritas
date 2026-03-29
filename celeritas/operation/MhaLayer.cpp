#include "MhaLayer.h"

#include <algorithm>
#include <stdexcept>

#include "celeritas/kvCache/kvCacheMgr.h"

namespace eCEL {

template <typename T>
void MhaLayer<T>::forward(const ForwardContext& ctx,
                          TensorListView<T> inputs,
                          eUTIL::Tensor<T>& output) {
    Layer<T>::requireInputCount(inputs, 1, "MhaLayer");

    if (ctx.kv_cache == nullptr) {
        throw std::invalid_argument("MhaLayer requires kv_cache in ForwardContext");
    }

    const eUTIL::Tensor<T>& query = inputs[0];
    validateTensorShapes(query, output);

    const KVCacheView* cacheView = ctx.kv_cache;
    if (cacheView->k_ptr == nullptr || cacheView->v_ptr == nullptr) {
        throw std::invalid_argument("MhaLayer requires valid KV cache pointers");
    }

    const std::size_t activeSeqLen =
        ctx.kv_len > 0 ? static_cast<std::size_t>(ctx.kv_len) : cacheView->seq_len;
    if (activeSeqLen == 0) {
        throw std::invalid_argument("MhaLayer requires kv_len greater than 0");
    }
    if (activeSeqLen > static_cast<std::size_t>(m_seqLen)) {
        throw std::out_of_range("MhaLayer kv_len exceeds configured seq_len");
    }

    eUTIL::Tensor<T> keyCacheTensor(
        this->m_device,
        {static_cast<std::size_t>(m_seqLen), static_cast<std::size_t>(m_kvDim)},
        reinterpret_cast<T*>(cacheView->k_ptr),
        true);
    eUTIL::Tensor<T> valueCacheTensor(
        this->m_device,
        {static_cast<std::size_t>(m_seqLen), static_cast<std::size_t>(m_kvDim)},
        reinterpret_cast<T*>(cacheView->v_ptr),
        true);
    eUTIL::Tensor<T> scoreTensor(
        this->m_device,
        static_cast<std::size_t>(m_headNum),
        static_cast<std::size_t>(m_seqLen));
    if (this->m_device == eUTIL::DeviceType::kCpu) {
        std::fill_n(output.data(), output.size(), static_cast<T>(0));
    }

    auto kernel = KernelFactory::getMhaKernel();
    kernel(static_cast<int32_t>(activeSeqLen - 1),
           m_headNum,
           0,
           m_seqLen,
           m_kvDim,
           m_kvMul,
           m_headSize,
           query,
           keyCacheTensor,
           valueCacheTensor,
           scoreTensor,
           output,
           ctx.cuda_config);
}

template <typename T>
void MhaLayer<T>::validateTensorShapes(const eUTIL::Tensor<T>& query,
                                       const eUTIL::Tensor<T>& output) const {
    const std::size_t expectedSize =
        static_cast<std::size_t>(m_headNum) * static_cast<std::size_t>(m_headSize);
    if (query.size() != expectedSize) {
        throw std::invalid_argument("MhaLayer query tensor shape does not match configured heads");
    }
    if (output.size() != expectedSize) {
        throw std::invalid_argument("MhaLayer output tensor shape does not match configured heads");
    }
}

template class MhaLayer<float>;

}  // namespace eCEL
