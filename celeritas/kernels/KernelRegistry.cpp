#include "cpu/add_kernel.h"
#include "gpu/add.cuh"
#include "cpu/emb_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "gpu/rmsnorm_kernel.cuh"
#include "KernelRegistry.h"

namespace eCEL {

KernelRegistry::KernelRegistry() {
    initRegistryTable();
}

void KernelRegistry::initRegistryTable()
{
    // ADD
    registerKernel<AddKernelFn<int>>(OpType::kAdd,
                                     eUTIL::DeviceType::kCpu,
                                     eUTIL::DType::kInt32,
                                     add_kernel_cpu<int>);
    registerKernel<AddKernelFn<float>>(OpType::kAdd,
                                       eUTIL::DeviceType::kCpu,
                                       eUTIL::DType::kFloat32,
                                       add_kernel_cpu<float>);
    registerKernel<AddKernelFn<int>>(OpType::kAdd,
                                     eUTIL::DeviceType::kCuda,
                                     eUTIL::DType::kInt32,
                                     add_kernel_cu<int>);
    
    // Embedding 
    registerKernel<EmbKernelFn<float>>(OpType::kEmb,
                                eUTIL::DeviceType::kCpu,
                                eUTIL::DType::kFloat32,
                                embKernelCpu<float>);

    // RMSNorm
    registerKernel<RmsKernelFn<float>>(OpType::kRms,
                                eUTIL::DeviceType::kCpu,
                                eUTIL::DType::kFloat32,
                                rmsKernelCpu<float>);
    registerKernel<RmsKernelFn<float>>(OpType::kRms,
                                eUTIL::DeviceType::kCuda,
                                eUTIL::DType::kFloat32,
                                rmsKernelCu<float>);
}

}  // namespace eCEL
