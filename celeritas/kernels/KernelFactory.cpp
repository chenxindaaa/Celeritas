#include "KernelFactory.h"

#include "cpu/add_kernel.h"
#include "cpu/emb_kernel.h"
#include "gpu/add.cuh"

namespace eCEL {

AddDispatcher KernelFactory::getAddKernel() {
    ensureRegistryInitialized();
    return AddDispatcher();
}

EmbDispatcher KernelFactory::getEmbKernel() {
    ensureRegistryInitialized();
    return EmbDispatcher();
}
 
void KernelFactory::ensureRegistryInitialized() {
    static std::once_flag initFlag;
    std::call_once(initFlag, [] {
        auto& dispatcher = Dispatcher::getInstance();
        dispatcher.registerKernel<AddKernelFn<int>>(OpType::kAdd,
                                                    eUTIL::DeviceType::kCpu,
                                                    eUTIL::DType::kInt32,
                                                    add_kernel_cpu<int>);
        dispatcher.registerKernel<AddKernelFn<float>>(OpType::kAdd,
                                                    eUTIL::DeviceType::kCpu,
                                                    eUTIL::DType::kFloat32,
                                                    add_kernel_cpu<float>);
        dispatcher.registerKernel<AddKernelFn<int>>(OpType::kAdd,
                                                    eUTIL::DeviceType::kCuda,
                                                    eUTIL::DType::kInt32,
                                                    add_kernel_cu<int>);
        dispatcher.registerKernel<EmbKernelFn>(OpType::kEmb,
                                               eUTIL::DeviceType::kCpu,
                                               eUTIL::DType::kFloat32,
                                               embKernelCpu);
    });
}

}  // namespace eCEL
