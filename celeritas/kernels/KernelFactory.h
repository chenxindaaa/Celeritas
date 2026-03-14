#pragma once

#include <cstdint>
#include <stdexcept>

#include "baseutil/tensor/tensor.h"

namespace eCEL {

template<typename T>
using AddKernelFn = void (*)(const eUTIL::Tensor<T>& input1,
                             const eUTIL::Tensor<T>& input2,
                             eUTIL::Tensor<T>& output,
                             void* stream);

using EmbKernelFn = void (*)(const eUTIL::Tensor<int>& input,
                             const eUTIL::Tensor<float>& weight,
                             eUTIL::Tensor<float>& output,
                             int32_t vocabSize,
                             void* stream);

class KernelFactory final {
    public:
        KernelFactory() = delete;

        template <typename First, typename... Rest>
        static bool hasUnknownDevice(const eUTIL::Tensor<First>& first,
                                     const eUTIL::Tensor<Rest>&... rest) {
            return (first.device() == eUTIL::DeviceType::kUnknown) ||
                   ((rest.device() == eUTIL::DeviceType::kUnknown) || ...);
        }

        template <typename First, typename... Rest>
        static bool checkSameDevice(const eUTIL::Tensor<First>& first,
                                    const eUTIL::Tensor<Rest>&... rest) {
            return ((first.device() == rest.device()) && ...);
        }

        template <typename T>
        static AddKernelFn<T> getAddKernel(const eUTIL::Tensor<T>& input1,
                                           const eUTIL::Tensor<T>& input2,
                                           const eUTIL::Tensor<T>& output) {
            if (hasUnknownDevice(input1, input2, output)) {
                throw std::invalid_argument("add kernel does not support kUnknown device");
            }

            if (!checkSameDevice(input1, input2, output)) {
                throw std::invalid_argument(
                    "add kernel requires all tensors on the same device");
            }

            const auto kernel = getAddKernelByDevice<T>(input1.device());
            if (!kernel) {
                throw std::invalid_argument("Unsupported tensor device");
            }
            return kernel;
        }

        static EmbKernelFn getEmbKernel() {
            return &dispatchEmbKernel;
        }

    private:
        static void dispatchEmbKernel(const eUTIL::Tensor<int>& input,
                                      const eUTIL::Tensor<float>& weight,
                                      eUTIL::Tensor<float>& output,
                                      int32_t vocabSize,
                                      void* stream) {
            if (hasUnknownDevice(input, weight, output)) {
                throw std::invalid_argument("emb kernel does not support kUnknown device");
            }

            if (!checkSameDevice(input, weight, output)) {
                throw std::invalid_argument(
                    "emb kernel requires all tensors on the same device");
            }

            const auto kernel = getEmbKernelByDevice(input.device());
            if (!kernel) {
                throw std::invalid_argument("Unsupported tensor device");
            }
            kernel(input, weight, output, vocabSize, stream);
        }

        template <typename T>
        static AddKernelFn<T> getAddKernelByDevice(eUTIL::DeviceType) {
            return nullptr;
        }

        static EmbKernelFn getEmbKernelByDevice(eUTIL::DeviceType);
};

template <>
AddKernelFn<int>
KernelFactory::getAddKernelByDevice<int>(eUTIL::DeviceType device);

}  // namespace eCEL
