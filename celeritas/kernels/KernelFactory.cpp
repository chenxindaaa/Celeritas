#include "KernelFactory.h"

#include <mutex>

#include "cpu/add_kernel.h"
#include "gpu/add.cuh"

namespace eCEL {

AddDispatcher KernelFactory::getAddKernel() {
    ensureRegistryInitialized();
    return AddDispatcher();
}

void KernelFactory::ensureRegistryInitialized() {
    static std::once_flag initFlag;
    std::call_once(initFlag, [] {
        KernelRegistry::registerAddKernel<int>(eUTIL::DeviceType::kCpu, add_kernel_cpu<int>);
        KernelRegistry::registerAddKernel<int>(eUTIL::DeviceType::kCuda, add_kernel_cu<int>);
    });
}

}  // namespace eCEL
