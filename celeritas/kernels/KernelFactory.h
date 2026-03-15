#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "baseutil/tensor/tensor.h"

namespace eCEL {

template <typename T>
using AddKernelFn = void (*)(const eUTIL::Tensor<T>& input1,
                             const eUTIL::Tensor<T>& input2,
                             eUTIL::Tensor<T>& output,
                             void* stream);

class KernelRegistry {
    public:
        template <typename T>
        static void registerAddKernel(eUTIL::DeviceType device, AddKernelFn<T> fn) {
            addKernelRepo<T>()[device] = fn;
        }

        template <typename T>
        static AddKernelFn<T> dispatchAddKernel(eUTIL::DeviceType device) {
            auto& repo = addKernelRepo<T>();
            const auto it = repo.find(device);
            return it == repo.end() ? nullptr : it->second;
        }

    private:
        struct DeviceTypeHash {
            std::size_t operator()(eUTIL::DeviceType device) const noexcept {
                return static_cast<std::size_t>(static_cast<int>(device) + 1);
            }
        };

        template <typename T>
        static std::unordered_map<eUTIL::DeviceType, AddKernelFn<T>, DeviceTypeHash>& addKernelRepo() {
            static std::unordered_map<eUTIL::DeviceType, AddKernelFn<T>, DeviceTypeHash> repo;
            return repo;
        }
};

class AddDispatcher;

class KernelFactory final {
    public:
        KernelFactory() = delete;

        static AddDispatcher getAddKernel();

        template <typename T>
        static AddDispatcher getAddKernel(const eUTIL::Tensor<T>& input1,
                                          const eUTIL::Tensor<T>& input2,
                                          const eUTIL::Tensor<T>& output);

        template <typename FirstTensor, typename... RestTensors>
        static eUTIL::DeviceType checkSameDevice(const char* opName,
                                                 const FirstTensor& first,
                                                 const RestTensors&... rest) {
            static_assert(isTensorArg<FirstTensor>::value,
                          "checkSameDevice requires Tensor arguments");
            static_assert((isTensorArg<RestTensors>::value && ...),
                          "checkSameDevice requires Tensor arguments");

            const eUTIL::DeviceType baseDevice = first.device();
            if (eUTIL::IsUnknownDevice(baseDevice)) {
                throw std::invalid_argument(buildUnknownDeviceError(opName));
            }

            auto checkOne = [&](const auto& tensor) {
                if (eUTIL::IsUnknownDevice(tensor.device())) {
                    throw std::invalid_argument(buildUnknownDeviceError(opName));
                }
                if (tensor.device() != baseDevice) {
                    throw std::invalid_argument(buildDeviceMismatchError(opName));
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

        static std::string buildUnknownDeviceError(const char* opName) {
            return std::string(opName) + " received tensor on kUnknown device";
        }

        static std::string buildDeviceMismatchError(const char* opName) {
            return std::string(opName) + " requires all tensors on the same device";
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
                KernelFactory::checkSameDevice("add", input1, input2, output);

            auto kernel = KernelRegistry::dispatchAddKernel<T>(device);
            if (kernel == nullptr) {
                throw std::invalid_argument(
                    "add kernel is not registered for this tensor type/device");
            }

            kernel(input1, input2, output, stream);
        }
};

template <typename T>
AddDispatcher KernelFactory::getAddKernel(const eUTIL::Tensor<T>& input1,
                                          const eUTIL::Tensor<T>& input2,
                                          const eUTIL::Tensor<T>& output) {
    (void)checkSameDevice("add", input1, input2, output);
    return getAddKernel();
}

}  // namespace eCEL
