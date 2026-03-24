#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

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

template <typename Tin, typename Tw = Tin, typename Ts = float, typename Tout = Tin>
using MatmulKernelFn = void (*)(const eUTIL::Tensor<Tin>& input,
                                const eUTIL::Tensor<Tw>& weight,
                                const eUTIL::Tensor<Ts>& scaler,
                                eUTIL::Tensor<Tout>& output,
                                int32_t group_size,
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

inline std::string dtypeName(eUTIL::DType dtype) {
    switch (dtype) {
        case eUTIL::DType::kInt8:
            return "int8";
        case eUTIL::DType::kFloat16:
            return "float16";
        case eUTIL::DType::kFloat32:
            return "float32";
        case eUTIL::DType::kUnknown:
        case eUTIL::DType::kNumDTypes:
        default:
            return "unknown";
    }
}

inline std::string tensorSignatureToString(const std::vector<eUTIL::DType>& dtypes) {
    std::string signature = "[";
    for (std::size_t i = 0; i < dtypes.size(); ++i) {
        if (i > 0) {
            signature += ", ";
        }
        signature += dtypeName(dtypes[i]);
    }
    signature += "]";
    return signature;
}

template <typename... Ts>
std::vector<eUTIL::DType> makeTensorSignature() {
    return {eUTIL::DTypeTrait<Ts>::kValue...};
}

struct KernelKey {
    OpType op = OpType::kUnknown;
    eUTIL::DeviceType device = eUTIL::DeviceType::kUnknown;
    std::vector<eUTIL::DType> tensorDtypes;

    bool operator==(const KernelKey& other) const {
        return op == other.op &&
               device == other.device &&
               tensorDtypes == other.tensorDtypes;
    }
};

struct KernelKeyHash {
    std::size_t operator()(const KernelKey& key) const {
        std::size_t seed = std::hash<int>{}(static_cast<int>(key.op));
        seed ^= std::hash<int>{}(static_cast<int>(key.device)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        for (const auto dtype : key.tensorDtypes) {
            seed ^= std::hash<int>{}(static_cast<int>(dtype)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class KernelRegistry final : public Singleton<KernelRegistry> {
friend class Singleton<KernelRegistry>;
friend class Dispatcher;
private:
    using FnPtr = void (*)();
    KernelRegistry();
    template <typename KernelFn>
    void registerKernel(OpType opType,
                        eUTIL::DeviceType device,
                        std::vector<eUTIL::DType> tensorDtypes,
                        KernelFn fn) {
        m_regTable[KernelKey{opType, device, std::move(tensorDtypes)}] = reinterpret_cast<FnPtr>(fn);
    }

    template <typename KernelFn>
    KernelFn lookup(OpType opType,
                    eUTIL::DeviceType device,
                    const std::vector<eUTIL::DType>& tensorDtypes) const {
        const auto it = m_regTable.find(KernelKey{opType, device, tensorDtypes});
        if (it == m_regTable.end() || it->second == nullptr) {
            std::string log = opTypeName(opType) + " kernel is not registered for device=" +
                              std::to_string(static_cast<int>(device)) +
                              " tensor_dtypes=" + tensorSignatureToString(tensorDtypes);
            throw std::invalid_argument(log);
        }
        return reinterpret_cast<KernelFn>(it->second);
    }
    void initRegistryTable();

private:
    std::unordered_map<KernelKey, FnPtr, KernelKeyHash> m_regTable{};
};

}  // namespace eCEL
