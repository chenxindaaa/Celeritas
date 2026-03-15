#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <type_traits>

#include "baseutil/tensor/tensor.h"
#include "baseutil/designPattren/Singleton.h"

namespace eCEL {

enum class OpType {
    kUnknown,
    kAdd,
    kEmb,
    kMatmul,
    kMha,
    kNumOpTypes,
};

inline const char* opTypeName(OpType opType) {
    constexpr const char* kOpNames[] = {
        "unknown",
        "add",
        "emb",
        "matmul",
        "mha",
    };
    const int opIndex = static_cast<int>(opType);
    if (opIndex < 0 || opIndex >= static_cast<int>(OpType::kNumOpTypes)) {
        return "unknown";
    }
    return kOpNames[opIndex];
}

inline std::string kernelNotRegisteredError(OpType opType) {
    return std::string(opTypeName(opType))
           + " kernel is not registered for this tensor type/device";
}

inline std::string unsupportedDTypeError(OpType opType) {
    return std::string(opTypeName(opType)) + " received unsupported tensor dtype";
}

inline std::string kernelTypeMismatchError(OpType opType) {
    return std::string(opTypeName(opType)) + " kernel signature mismatch";
}

template <typename T>
using AddKernelFn = void (*)(const eUTIL::Tensor<T>& input1,
                             const eUTIL::Tensor<T>& input2,
                             eUTIL::Tensor<T>& output,
                             void* stream);

using EmbKernelFn = void (*)(const eUTIL::Tensor<int>& input,
                             const eUTIL::Tensor<float>& weight,
                             eUTIL::Tensor<float>& output,
                             int32_t vocabSize,
                             void* stream);

class KernelFunction {
    using FnPtr = void(*)();
    public:
        KernelFunction() : fnPtr_(nullptr), fnType_(typeid(void)) {}

        template <typename KernelFn>
        static KernelFunction make(KernelFn fn) {
            static_assert(std::is_pointer_v<KernelFn>,
                          "KernelFunction::make requires a function pointer type");
            static_assert(std::is_function_v<std::remove_pointer_t<KernelFn>>,
                          "KernelFunction::make requires a function pointer type");

            return KernelFunction(reinterpret_cast<FnPtr>(fn), std::type_index(typeid(KernelFn)));
        }

        bool isValid() const {
            return fnPtr_ != nullptr;
        }

        template <typename KernelFn>
        KernelFn typed(OpType opType) const {
            static_assert(std::is_pointer_v<KernelFn>,
                          "KernelFunction::typed requires a function pointer type");
            static_assert(std::is_function_v<std::remove_pointer_t<KernelFn>>,
                          "KernelFunction::typed requires a function pointer type");

            if (!isValid()) {
                return nullptr;
            }

            if (fnType_ != std::type_index(typeid(KernelFn))) {
                throw std::invalid_argument(kernelTypeMismatchError(opType));
            }

            return reinterpret_cast<KernelFn>(fnPtr_);
        }

    private:
        KernelFunction(FnPtr fnPtr, std::type_index fnType) : fnPtr_(fnPtr), fnType_(fnType) {}

        FnPtr fnPtr_;
        std::type_index fnType_;
};

class Dispatcher final : public Singleton<Dispatcher> {
    friend class Singleton<Dispatcher>;

public:
    template <typename KernelFn>
    void registerKernel(OpType opType,
                        eUTIL::DeviceType device,
                        eUTIL::DType dtype,
                        KernelFn fn) {
        dispatchTable_[opIndex(opType)][deviceIndex(device)][dtypeIndex(dtype)] =
            KernelFunction::make(fn);
    }

    KernelFunction lookupKernel(OpType opType,
                                eUTIL::DeviceType device,
                                eUTIL::DType dtype) const {
        return dispatchTable_[opIndex(opType)][deviceIndex(device)][dtypeIndex(dtype)];
    }

private:
    Dispatcher() = default;

    static constexpr std::size_t kOpTypeCount =
        static_cast<std::size_t>(OpType::kNumOpTypes);
    static constexpr std::size_t kDeviceCount =
        static_cast<std::size_t>(eUTIL::DeviceType::kNumDeviceTypes);
    static constexpr std::size_t kDTypeCount =
        static_cast<std::size_t>(eUTIL::DType::kNumDTypes);

    static std::size_t opIndex(OpType opType) {
        const int value = static_cast<int>(opType);
        if (value <= static_cast<int>(OpType::kUnknown)
            || value >= static_cast<int>(OpType::kNumOpTypes)) {
            throw std::invalid_argument("invalid OpType for dispatcher table");
        }
        return static_cast<std::size_t>(value);
    }

    static std::size_t deviceIndex(eUTIL::DeviceType device) {
        const int value = static_cast<int>(device);
        if (value <= static_cast<int>(eUTIL::DeviceType::kUnknown)
            || value >= static_cast<int>(eUTIL::DeviceType::kNumDeviceTypes)) {
            throw std::invalid_argument("invalid DeviceType for dispatcher table");
        }
        return static_cast<std::size_t>(value);
    }

    static std::size_t dtypeIndex(eUTIL::DType dtype) {
        const int value = static_cast<int>(dtype);
        if (value <= static_cast<int>(eUTIL::DType::kUnknown)
            || value >= static_cast<int>(eUTIL::DType::kNumDTypes)) {
            throw std::invalid_argument("invalid DType for dispatcher table");
        }
        return static_cast<std::size_t>(value);
    }

    std::array<std::array<std::array<KernelFunction, kDTypeCount>, kDeviceCount>,
                kOpTypeCount> dispatchTable_{};
};

class AddDispatcher;
class EmbDispatcher;

class KernelFactory final {
    public:
        KernelFactory() = delete;

        static AddDispatcher getAddKernel();
        static EmbDispatcher getEmbKernel();

        template <typename T>
        static AddDispatcher getAddKernel(const eUTIL::Tensor<T>& input1,
                                          const eUTIL::Tensor<T>& input2,
                                          const eUTIL::Tensor<T>& output);
        static EmbDispatcher getEmbKernel(const eUTIL::Tensor<int>& input,
                                          const eUTIL::Tensor<float>& weight,
                                          const eUTIL::Tensor<float>& output);

        template <typename FirstTensor, typename... RestTensors>
        static eUTIL::DeviceType checkSameDevice(OpType opType,
                                                 const FirstTensor& first,
                                                 const RestTensors&... rest) {
            static_assert(isTensorArg<FirstTensor>::value,
                          "checkSameDevice requires Tensor arguments");
            static_assert((isTensorArg<RestTensors>::value && ...),
                          "checkSameDevice requires Tensor arguments");

            const eUTIL::DeviceType baseDevice = first.device();
            if (eUTIL::IsUnknownDevice(baseDevice)) {
                throw std::invalid_argument(buildUnknownDeviceError(opType));
            }

            auto checkOne = [&](const auto& tensor) {
                if (eUTIL::IsUnknownDevice(tensor.device())) {
                    throw std::invalid_argument(buildUnknownDeviceError(opType));
                }
                if (tensor.device() != baseDevice) {
                    throw std::invalid_argument(buildDeviceMismatchError(opType));
                }
            };

            (checkOne(rest), ...);
            return baseDevice;
        }

        static void ensureRegistryInitialized();

    private:
        template <typename T>
        struct isTensor : std::false_type {};

        template <typename T>
        struct isTensor<eUTIL::Tensor<T>> : std::true_type {};

        template <typename T>
        struct isTensorArg : isTensor<std::remove_cv_t<std::remove_reference_t<T>>> {};

        static std::string buildUnknownDeviceError(OpType opType) {
            return std::string(opTypeName(opType)) + " received tensor on kUnknown device";
        }

        static std::string buildDeviceMismatchError(OpType opType) {
            return std::string(opTypeName(opType))
                   + " requires all tensors on the same device";
        }
};

class AddDispatcher {
    public:
        template <typename T>
        void operator()(const eUTIL::Tensor<T>& input1,
                        const eUTIL::Tensor<T>& input2,
                        eUTIL::Tensor<T>& output,
                        void* stream = nullptr) const {
            const eUTIL::DeviceType device =
                KernelFactory::checkSameDevice(OpType::kAdd, input1, input2, output);

            constexpr eUTIL::DType dtype = eUTIL::DTypeTrait<T>::kValue;
            if (dtype == eUTIL::DType::kUnknown) {
                throw std::invalid_argument(unsupportedDTypeError(OpType::kAdd));
            }

            const auto boxedKernel =
                Dispatcher::getInstance().lookupKernel(OpType::kAdd, device, dtype);
            auto kernel = boxedKernel.typed<AddKernelFn<T>>(OpType::kAdd);
            if (kernel == nullptr) {
                throw std::invalid_argument(kernelNotRegisteredError(OpType::kAdd));
            }

            kernel(input1, input2, output, stream);
        }
};

class EmbDispatcher {
    public:
        void operator()(const eUTIL::Tensor<int>& input,
                        const eUTIL::Tensor<float>& weight,
                        eUTIL::Tensor<float>& output,
                        int32_t vocabSize,
                        void* stream = nullptr) const {
            const eUTIL::DeviceType device =
                KernelFactory::checkSameDevice(OpType::kEmb, input, weight, output);

            const eUTIL::DType dtype = output.dtype();
            if (dtype == eUTIL::DType::kUnknown) {
                throw std::invalid_argument(unsupportedDTypeError(OpType::kEmb));
            }

            const auto boxedKernel =
                Dispatcher::getInstance().lookupKernel(OpType::kEmb, device, dtype);
            auto kernel = boxedKernel.typed<EmbKernelFn>(OpType::kEmb);
            if (kernel == nullptr) {
                throw std::invalid_argument(kernelNotRegisteredError(OpType::kEmb));
            }

            kernel(input, weight, output, vocabSize, stream);
        }
};

template <typename T>
AddDispatcher KernelFactory::getAddKernel(const eUTIL::Tensor<T>& input1,
                                          const eUTIL::Tensor<T>& input2,
                                          const eUTIL::Tensor<T>& output) {
    (void)checkSameDevice(OpType::kAdd, input1, input2, output);
    return getAddKernel();
}

inline EmbDispatcher KernelFactory::getEmbKernel(const eUTIL::Tensor<int>& input,
                                                 const eUTIL::Tensor<float>& weight,
                                                 const eUTIL::Tensor<float>& output) {
    (void)checkSameDevice(OpType::kEmb, input, weight, output);
    return getEmbKernel();
}

}  // namespace eCEL
