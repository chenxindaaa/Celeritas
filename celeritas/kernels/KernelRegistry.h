#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <typeindex>
#include <type_traits>
#include <string>

#include "udm/core/Tensor.h"
#include "udm/designPattren/Singleton.h"
#include "udm/common/cudaConfig.h"

namespace eCEL {

enum class OpType {
    kUnknown,
    kAdd,
    kEmb,
    kRms,
    kMatmul,
    kMha,
    kNumOpTypes,
};

inline std::string opTypeName(OpType opType) {
    switch (opType) {
        case OpType::kAdd:
            return "add";
        case OpType::kEmb:
            return "emb";
        case OpType::kRms:
            return "rms";
        case OpType::kMatmul:
            return "matmul";
        case OpType::kMha:
            return "mha";
        case OpType::kUnknown:
        case OpType::kNumOpTypes:
        default:
            return "unknown";
    }
}

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

template <typename T>
using RmsKernelFn = void (*)(const eUTIL::Tensor<T>& input,
                             const eUTIL::Tensor<T>& weight,
                             eUTIL::Tensor<T>& output,
                             void* stream);      
                             
template <typename T>
using MatmulKernelFn = void (*)(const eUTIL::Tensor<T>& input,
                                const eUTIL::Tensor<T>& weight,
                                eUTIL::Tensor<T>& output,
                                const float scale,
                                const eUTIL::CudaConfig* config);  

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

    std::array<std::array<std::array<FnPtr, (std::size_t)(eUTIL::DType::kNumDTypes)>,
                          (std::size_t)(eUTIL::DeviceType::kNumDeviceTypes)>,
               (std::size_t)(OpType::kNumOpTypes)>
        m_regTable{};
};

}  // namespace eCEL
