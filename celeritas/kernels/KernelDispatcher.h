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
                                m_device(eUTIL::DeviceType::kUnknown) {}

protected:
    template <typename FirstTensor, typename... RestTensors>
    void check(const FirstTensor& first,
               const RestTensors&... rest) const {
        m_device = first.device();
        if (eUTIL::isUnknownDevice(m_device)) {
            throw std::invalid_argument(
                std::string(opTypeName(m_opType)) + " received tensor on kUnknown device");
        }

        m_tensorDtypes.clear();
        auto collectOne = [&](const auto& t) {
            const auto dtype = t.dtype();
            if (dtype == eUTIL::DType::kUnknown) {
                throw std::invalid_argument(
                    std::string(opTypeName(m_opType)) + " received unsupported tensor dtype");
            }
            m_tensorDtypes.push_back(dtype);
        };
        collectOne(first);

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
            collectOne(t);
        };
        (checkOne(rest), ...);
    }

    template<typename KernelFn>
    KernelFn lookup() const
    {
        return  m_kernelRegistry.lookup<KernelFn>(m_opType, m_device, m_tensorDtypes);
    }

protected:
    static KernelRegistry& m_kernelRegistry;
    const OpType m_opType;
    mutable eUTIL::DeviceType m_device;
    mutable std::vector<eUTIL::DType> m_tensorDtypes;
};

class AddDispatcher final : public Dispatcher {
public:
    AddDispatcher(OpType opType) : Dispatcher(opType) {}
    template <typename T>
    void operator()(const eUTIL::Tensor<T>& input1,
                    const eUTIL::Tensor<T>& input2,
                    eUTIL::Tensor<T>& output,
                    void* stream = nullptr) const {
        check(input1, input2, output);
        auto kernel = lookup<AddKernelFn<T>>();
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
        check(input, weight, output);
        auto kernel = lookup<EmbKernelFn<T>>();
        kernel(input, weight, output, vocabSize, stream);
    }
};

class RmsDispatcher final : public Dispatcher {
public:
    RmsDispatcher(OpType opType) : Dispatcher(opType) {}
    template <typename T>
    void operator()(const eUTIL::Tensor<T>& input,
                    const eUTIL::Tensor<T>& weight,
                    eUTIL::Tensor<T>& output,
                    void* stream = nullptr) const {
        check(input, weight, output);
        auto kernel = lookup<RmsKernelFn<T>>();
        kernel(input, weight, output, stream);
    }
};

class MatmulDispatcher final : public Dispatcher {
public:
    MatmulDispatcher(OpType opType) : Dispatcher(opType) {}
    template <typename Tin, typename Tw = Tin, typename Ts = float, typename Tout = Tin>
    void operator()(const eUTIL::Tensor<Tin>& input,
                    const eUTIL::Tensor<Tw>& weight,
                    const eUTIL::Tensor<Ts>& scaler,
                    eUTIL::Tensor<Tout>& output,
                    int32_t group_size,
                    const eUTIL::CudaConfig* config = nullptr) const {
        check(input, weight, scaler, output);
        auto kernel = lookup<MatmulKernelFn<Tin, Tw, Ts, Tout>>();
        kernel(input, weight, scaler, output, group_size, config);
    }
};

class SwigluDispatcher final : public Dispatcher {
public:
    SwigluDispatcher(OpType opType) : Dispatcher(opType) {}
    template <typename T>
    void operator()(const eUTIL::Tensor<T>& input1,
                    const eUTIL::Tensor<T>& input2,
                    eUTIL::Tensor<T>& output,
                    void* stream = nullptr) const {
        check(input1, input2, output);
        auto kernel = lookup<SwigluKernelFn<T>>();
        kernel(input1, input2, output, stream);
    }
};

class SoftmaxDispatcher final : public Dispatcher {
public:
    SoftmaxDispatcher(OpType opType) : Dispatcher(opType) {}
    template <typename T>
    void operator()(eUTIL::Tensor<T>& input,
                    void* stream = nullptr) const {
        check(input);
        auto kernel = lookup<SoftmaxKernelFn<T>>();
        kernel(input, stream);
    }
};

class ScalesumDispatcher final : public Dispatcher {
public:
    ScalesumDispatcher(OpType opType) : Dispatcher(opType) {}
    template <typename T>
    void operator()(const eUTIL::Tensor<T>& value, 
                    const eUTIL::Tensor<T>& scale,
                    eUTIL::Tensor<T>& output, 
                    int pos, int size, int stride,
                    void* stream = nullptr) const {
        check(value, scale, output);
        auto kernel = lookup<ScalesumKernelFn<T>>();
        kernel(value, scale, output, pos, size, stride, stream);
    }
};

class MhaDispatcher final : public Dispatcher {
public:
    MhaDispatcher(OpType opType) : Dispatcher(opType) {}
    template <typename T>
    void operator()(int32_t pos, int32_t head_num, 
                    int32_t layer_index, int32_t seq_len, 
                    int32_t kv_dim, int32_t kv_mul, int32_t head_size, 
                    const eUTIL::Tensor<T>& query_tensor, 
                    const eUTIL::Tensor<T>& key_cache_tensor, 
                    const eUTIL::Tensor<T>& value_cache_tensor,
                    eUTIL::Tensor<T>& score_tensor,
                    eUTIL::Tensor<T>& mha_out,
                    const eUTIL::CudaConfig* config = nullptr) const {
        check(query_tensor, key_cache_tensor, value_cache_tensor, score_tensor, mha_out);
        auto kernel = lookup<MhaKernelFn<T>>();
        kernel(pos, head_num, layer_index, seq_len, kv_dim, kv_mul, head_size, 
               query_tensor, key_cache_tensor, value_cache_tensor, score_tensor, mha_out, config);
    }
};

}  // namespace eCEL
