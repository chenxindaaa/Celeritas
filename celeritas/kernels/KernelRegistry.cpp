#include "KernelRegistry.h"

#include "cpu/add_kernel.h"
#include "cpu/emb_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/softmax_kernel.h"
#include "cpu/swiglu_kernel.h"
#include "cpu/scale_sum_kernel.h"
#include "cpu/mha_kernel.h"
#include "gpu/add.cuh"
#include "gpu/emb_kernel.cuh"
#include "gpu/matmul_kernel.cuh"
#include "gpu/rmsnorm_kernel.cuh"
#include "gpu/swiglu_kernel.cuh"
#include "gpu/mha_kernel.cuh"

namespace eCEL {

KernelRegistry::KernelRegistry() {
    initRegistryTable();
}

void KernelRegistry::initRegistryTable() {
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
    registerKernel<EmbKernelFn<float>>(OpType::kEmb,
                                       eUTIL::DeviceType::kCuda,
                                       eUTIL::DType::kFloat32,
                                       embKernelCu<float>);

    // RMSNorm
    registerKernel<RmsKernelFn<float>>(OpType::kRms,
                                       eUTIL::DeviceType::kCpu,
                                       eUTIL::DType::kFloat32,
                                       rmsKernelCpu<float>);
    registerKernel<RmsKernelFn<float>>(OpType::kRms,
                                       eUTIL::DeviceType::kCuda,
                                       eUTIL::DType::kFloat32,
                                       rmsKernelCu<float>);

    // Matmul
    registerKernel<MatmulKernelFn<float>>(OpType::kMatmul,
                                          eUTIL::DeviceType::kCpu,
                                          eUTIL::DType::kFloat32,
                                          matmulKernelCpu<float>);
    registerKernel<MatmulKernelFn<float>>(OpType::kMatmul,
                                          eUTIL::DeviceType::kCuda,
                                          eUTIL::DType::kFloat32,
                                          matmulKernelCu<float>);

    // Swiglu
    registerKernel<SwigluKernelFn<float>>(OpType::kSwiglu,
                                          eUTIL::DeviceType::kCpu,
                                          eUTIL::DType::kFloat32,
                                          swigluKernelCpu<float>);
    registerKernel<SwigluKernelFn<float>>(OpType::kSwiglu,
                                          eUTIL::DeviceType::kCuda,
                                          eUTIL::DType::kFloat32,
                                          swigluKernelCu<float>);

    // Softmax
    registerKernel<SoftmaxKernelFn<float>>(OpType::kSoftmax,
                                          eUTIL::DeviceType::kCpu,
                                          eUTIL::DType::kFloat32,
                                          softmaxInplaceCpu<float>);
    
    // Scalesum
    registerKernel<ScalesumKernelFn<float>>(OpType::kScalesum,
                                          eUTIL::DeviceType::kCpu,
                                          eUTIL::DType::kFloat32,
                                          scalesumKernelCpu<float>);
    
    // Mha
    registerKernel<MhaKernelFn<float>>(OpType::kMha,
                                       eUTIL::DeviceType::kCpu,
                                       eUTIL::DType::kFloat32,
                                       mhaKernelCpu<float>);
    registerKernel<MhaKernelFn<float>>(OpType::kMha,
                                       eUTIL::DeviceType::kCuda,
                                       eUTIL::DType::kFloat32,
                                       mhaKernelCu<float>);                                                                                               
}

}  // namespace eCEL
