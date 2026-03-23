#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <typeindex>

#include "udm/common/cudaConfig.h"
#include "udm/core/Tensor.h"
#include "udm/designPattren/Singleton.h"

namespace eCEL {

class Dispatcher;

enum class OpType {
    kUnknown,
    kAdd,
    kEmb,
    kRms,
    kMatmul,
    kSwiglu,
    kSoftmax,
    kScalesum,
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
        case OpType::kSwiglu:
            return "swiglu";
        case OpType::kSoftmax:
            return "softmax";
        case OpType::kScalesum:
            return "scalesum";
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

template <typename T>
using SwigluKernelFn = void (*)(const eUTIL::Tensor<T>& input1,
                                const eUTIL::Tensor<T>& input2,
                                eUTIL::Tensor<T>& output,
                                void* stream);

template <typename T>
using SoftmaxKernelFn = void (*)(eUTIL::Tensor<T>& input,
                                 void* stream);

template <typename T>
using ScalesumKernelFn = void (*)(const eUTIL::Tensor<T>& value, 
                                  const eUTIL::Tensor<T>& scale,
                                  eUTIL::Tensor<T>& output, 
                                  int pos, int size, int stride,
                                  void* stream);

template <typename T>
using MhaKernelFn = void (*)(int32_t pos, int32_t head_num, 
                             int32_t layer_index, int32_t seq_len, 
                             int32_t kv_dim, int32_t kv_mul, int32_t head_size, 
                             const eUTIL::Tensor<T>& query_tensor, 
                             const eUTIL::Tensor<T>& key_cache_tensor, 
                             const eUTIL::Tensor<T>& value_cache_tensor,
                             eUTIL::Tensor<T>& score_tensor,
                             eUTIL::Tensor<T>& mha_out,
                             const eUTIL::CudaConfig* config);

class KernelRegistry final : public Singleton<KernelRegistry> {
friend class Singleton<KernelRegistry>;
friend class Dispatcher;
private:
    using FnPtr = void (*)();
    KernelRegistry();
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
    void initRegistryTable();

private:
    std::array<std::array<std::array<FnPtr, (std::size_t)(eUTIL::DType::kNumDTypes)>,
                          (std::size_t)(eUTIL::DeviceType::kNumDeviceTypes)>,
               (std::size_t)(OpType::kNumOpTypes)>
        m_regTable{};
};

}  // namespace eCEL
