#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>

#include "KernelRegistry.h"

namespace eCEL {

class Dispatcher {
public:
    virtual ~Dispatcher() = default;
    Dispatcher(OpType opType) : m_opType(opType),
                                m_device(eUTIL::DeviceType::kUnknown),
                                m_dtype(eUTIL::DType::kUnknown) {}

protected:
    template <typename T, typename FirstTensor, typename... RestTensors>
    void check(const FirstTensor& first,
               const RestTensors&... rest) const {
        // Check data type
        m_dtype = eUTIL::DTypeTrait<T>::kValue;
        if (m_dtype == eUTIL::DType::kUnknown) {
            throw std::invalid_argument(
                std::string(opTypeName(m_opType)) + " received unsupported tensor dtype");
        }

        // Check devices
        m_device = first.device();
        if (eUTIL::isUnknownDevice(m_device)) {
            throw std::invalid_argument(
                std::string(opTypeName(m_opType)) + " received tensor on kUnknown device");
        }
        auto checkOne = [&](const auto& t) {
            const auto dev = t.device();
            if (eUTIL::isUnknownDevice(dev)) {
                throw std::invalid_argument(
                    std::string(opTypeName(m_opType)) + " received tensor on kUnknown device");
            }
            if (dev != m_device) {
                throw std::invalid_argument(
                    std::string(opTypeName(m_opType)) + " requires all tensors on the same device");
            }
        };
        (checkOne(rest), ...);
    }

protected:
    OpType m_opType;
    mutable eUTIL::DeviceType m_device;
    mutable eUTIL::DType m_dtype;
};

class AddDispatcher final : public Dispatcher {
public:
    AddDispatcher(OpType opType) : Dispatcher(opType) {}
    template <typename T>
    void operator()(const eUTIL::Tensor<T>& input1,
                    const eUTIL::Tensor<T>& input2,
                    eUTIL::Tensor<T>& output,
                    void* stream = nullptr) const {
        check<T>(input1, input2, output);
        auto kernel = KernelRegistry::getInstance().lookup<AddKernelFn<T>>(m_opType, m_device, m_dtype);
        kernel(input1, input2, output, stream);
    }
};

class EmbDispatcher final : public Dispatcher {
public:
    EmbDispatcher(OpType opType) : Dispatcher(opType) {}
    template <typename T>
    void operator()(const eUTIL::Tensor<T>& input,
                    const eUTIL::Tensor<T>& weight,
                    eUTIL::Tensor<T>& output,
                    int32_t vocabSize,
                    void* stream = nullptr) const {
        check<T>(input, weight, output);
        auto kernel = KernelRegistry::getInstance().lookup<EmbKernelFn<T>>(m_opType, m_device, m_dtype);
        kernel(input, weight, output, vocabSize, stream);
    }
};


}  // namespace eCEL
