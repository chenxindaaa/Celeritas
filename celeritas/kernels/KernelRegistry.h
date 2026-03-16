#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <typeindex>
#include <type_traits>
#include <string>

#include "OpType.h"
#include "udm/core/Tensor.h"
#include "udm/designPattren/Singleton.h"

namespace eCEL {

template <typename T>
using AddKernelFn = void (*)(const eUTIL::Tensor<T>& input1,
                             const eUTIL::Tensor<T>& input2,
                             eUTIL::Tensor<T>& output,
                             void* stream);

template <typename T>
using EmbKernelFn = void (*)(const eUTIL::Tensor<T>& input,
                             const eUTIL::Tensor<T>& weight,
                             eUTIL::Tensor<T>& output,
                             int32_t vocabSize,
                             void* stream);

class KernelRegistry final : public Singleton<KernelRegistry> {
friend class Singleton<KernelRegistry>;
public:
    using FnPtr = void (*)();
    template <typename KernelFn>
    void registerKernel(OpType opType,
                        eUTIL::DeviceType device,
                        eUTIL::DType dtype,
                        KernelFn fn) {
        m_regTable[(std::size_t)(opType)][(std::size_t)(device)][(std::size_t)(dtype)] =
            reinterpret_cast<FnPtr>(fn);
    }

    template <typename KernelFn>
    KernelFn lookup(OpType opType,
                 eUTIL::DeviceType device,
                 eUTIL::DType dtype) const {
        auto fnPtr = m_regTable[(std::size_t)(opType)][(std::size_t)(device)][(std::size_t)(dtype)];
        if (fnPtr == nullptr) {
            std::string log = opTypeName(opType) + " kernel is not registered for this tensor type/device";
            throw std::invalid_argument(log);
        }
        return reinterpret_cast<KernelFn>(fnPtr);
    }

private:
    KernelRegistry();
    void initRegistryTable();

    std::array<std::array<std::array<FnPtr, (std::size_t)(OpType::kNumOpTypes)>, 
                                            (std::size_t)(eUTIL::DeviceType::kNumDeviceTypes)>,
                                            (std::size_t)(eUTIL::DType::kNumDTypes)>
        m_regTable{};
};

}  // namespace eCEL
