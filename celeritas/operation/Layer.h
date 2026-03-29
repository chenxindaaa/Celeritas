#pragma once

// Defines the shared layer interface, forward context, and lightweight tensor input view.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "udm/common/cudaConfig.h"
#include "udm/core/Tensor.h"

namespace eCEL {

class Workspace {
public:
    virtual ~Workspace() = default;
};

class KVCacheView;
class KVCacheManager;
class RopeCacheView;

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
    KVCacheManager* kv_cache_manager = nullptr;
    const RopeCacheView* rope_cache = nullptr;
};

template <typename T>
class TensorListView {
public:
    TensorListView(const eUTIL::Tensor<T>* const* data, std::size_t size)
        : m_data(data),
          m_size(size) {}

    const eUTIL::Tensor<T>& operator[](std::size_t index) const { return *m_data[index]; }

    std::size_t size() const { return m_size; }

private:
    const eUTIL::Tensor<T>* const* m_data;
    std::size_t m_size;
};

template <typename T>
class Layer {
public:
    Layer() noexcept : m_device(eUTIL::DeviceType::kCpu) {}
    explicit Layer(eUTIL::DeviceType device) : m_device(device) {}
    virtual ~Layer() = default;

    virtual void forward(const ForwardContext& ctx,
                         TensorListView<T> inputs,
                         eUTIL::Tensor<T>& output) = 0;

    void forward(const ForwardContext& ctx,
                 const eUTIL::Tensor<T>& input,
                 eUTIL::Tensor<T>& output) {
        const eUTIL::Tensor<T>* inputList[] = {&input};
        forward(ctx, TensorListView<T>(inputList, 1), output);
    }

    eUTIL::DeviceType device() const { return m_device; }

    virtual void to(eUTIL::DeviceType device) { m_device = device; }

    void cpu() { to(eUTIL::DeviceType::kCpu); }

    void cuda() { to(eUTIL::DeviceType::kCuda); }

protected:
    // Validates the number of runtime input tensors for each layer.
    static void requireInputCount(TensorListView<T> inputs,
                                  std::size_t expectedCount,
                                  const char* layerName) {
        if (inputs.size() != expectedCount) {
            throw std::invalid_argument(
                std::string(layerName) + " expects " + std::to_string(expectedCount) +
                " input tensor(s), got " + std::to_string(inputs.size()));
        }
    }

    eUTIL::DeviceType m_device;
};

template <typename T>
class Parameter {
public:
    Parameter()
        : m_parameter(eUTIL::DeviceType::kCpu) {}

    explicit Parameter(eUTIL::Tensor<T> parameter)
        : m_parameter(std::move(parameter)) {}

    const eUTIL::Tensor<T>& view() const { return m_parameter; }
    eUTIL::Tensor<T>& view() { return m_parameter; }

    void to(eUTIL::DeviceType device) {
        if (device == eUTIL::DeviceType::kCuda) {
            m_parameter.cuda();
            return;
        }
        if (device == eUTIL::DeviceType::kCpu) {
            m_parameter.cpu();
            return;
        }
        throw std::invalid_argument("Parameter::to received unsupported device");
    }

    void cpu() { to(eUTIL::DeviceType::kCpu); }

    void cuda() { to(eUTIL::DeviceType::kCuda); }

private:
    eUTIL::Tensor<T> m_parameter;
};

}  // namespace eCEL
