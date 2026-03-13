#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "baseutil/tensor/tensor.h"

namespace eCEL {

enum class Status {
    kOk = 0,
    kInvalidArgument = 1,
    kRuntimeError = 2,
};

enum class LayerType {
    kUnknown = 0,
    kAdd = 1,
};

template <typename T>
class BaseLayer {
   public:
    using TensorType = eUTIL::Tensor<T>;
    using InputList = std::vector<std::reference_wrapper<const TensorType>>;

    explicit BaseLayer(eUTIL::DeviceType device, LayerType type,
                       std::string name = "")
        : m_device(device), m_type(type), m_name(std::move(name)) {}
    virtual ~BaseLayer() = default;

    BaseLayer(const BaseLayer&) = delete;
    BaseLayer& operator=(const BaseLayer&) = delete;

    // Unified forward API: arbitrary number of inputs + one output.
    virtual Status Forward(const InputList& inputs, TensorType& output) = 0;

    eUTIL::DeviceType device() const { return m_device; }
    LayerType type() const { return m_type; }
    const std::string& name() const { return m_name; }
    void set_name(const std::string& name) { m_name = name; }

   protected:
    eUTIL::DeviceType m_device;
    LayerType m_type;
    std::string m_name;
};

}  // namespace eCEL
