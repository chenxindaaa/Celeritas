#include "KernelFactory.h"

#include "cpu/add_kernel.h"
#include "cpu/emb_kernel.h"
#include "gpu/add.cuh"

namespace eCEL {

namespace {

class IKernelFactory {
    public:
        virtual ~IKernelFactory() = default;
        virtual AddKernelFn<int> createAddKernelInt() const = 0;
        virtual EmbKernelFn createEmbKernel() const = 0;
};

class CpuKernelFactory final : public IKernelFactory {
    public:
        AddKernelFn<int> createAddKernelInt() const override {
            return add_kernel_cpu<int>;
        }

        EmbKernelFn createEmbKernel() const override {
            return embKernelCpu;
        }
};

class CudaKernelFactory final : public IKernelFactory {
    public:
        AddKernelFn<int> createAddKernelInt() const override {
            return add_kernel_cu<int>;
        }

        EmbKernelFn createEmbKernel() const override {
            return nullptr;
        }
};

const IKernelFactory* resolveKernelFactory(eUTIL::DeviceType device) {
    static const CpuKernelFactory cpuFactory;
    static const CudaKernelFactory cudaFactory;

    switch (device) {
        case eUTIL::DeviceType::kUnknown:
            return nullptr;
        case eUTIL::DeviceType::kCpu:
            return &cpuFactory;
        case eUTIL::DeviceType::kCuda:
            return &cudaFactory;
        default:
            return nullptr;
    }
}

}  // namespace

template <>
AddKernelFn<int>
KernelFactory::getAddKernelByDevice<int>(eUTIL::DeviceType device) {
    if (device == eUTIL::DeviceType::kUnknown) {
        throw std::invalid_argument("kUnknown device is not supported by add kernel factory");
    }
    const auto* factory = resolveKernelFactory(device);
    return factory ? factory->createAddKernelInt() : nullptr;
}

EmbKernelFn
KernelFactory::getEmbKernelByDevice(eUTIL::DeviceType device) {
    if (device == eUTIL::DeviceType::kUnknown) {
        throw std::invalid_argument("kUnknown device is not supported by emb kernel factory");
    }
    const auto* factory = resolveKernelFactory(device);
    return factory ? factory->createEmbKernel() : nullptr;
}

}  // namespace eCEL
