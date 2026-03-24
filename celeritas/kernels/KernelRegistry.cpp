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
    registerKernel<AddKernelFn<float>>(OpType::kAdd,
                                       eUTIL::DeviceType::kCpu,
                                       makeTensorSignature<float, float, float>(),
                                       add_kernel_cpu<float>);
    registerKernel<AddKernelFn<float>>(OpType::kAdd,
                                       eUTIL::DeviceType::kCuda,
                                       makeTensorSignature<float, float, float>(),
                                       add_kernel_cu<float>);
    registerKernel<AddKernelFn<int8_t>>(OpType::kAdd,
                                        eUTIL::DeviceType::kCuda,
                                        makeTensorSignature<int8_t, int8_t, int8_t>(),
                                        add_kernel_cu<int8_t>);

    // Embedding
    registerKernel<EmbKernelFn<float>>(OpType::kEmb,
                                       eUTIL::DeviceType::kCpu,
                                       makeTensorSignature<float, float, float>(),
                                       embKernelCpu<float>);
    registerKernel<EmbKernelFn<float>>(OpType::kEmb,
                                       eUTIL::DeviceType::kCuda,
                                       makeTensorSignature<float, float, float>(),
                                       embKernelCu<float>);

    // RMSNorm
    registerKernel<RmsKernelFn<float>>(OpType::kRms,
                                       eUTIL::DeviceType::kCpu,
                                       makeTensorSignature<float, float, float>(),
                                       rmsKernelCpu<float>);
    registerKernel<RmsKernelFn<float>>(OpType::kRms,
                                       eUTIL::DeviceType::kCuda,
                                       makeTensorSignature<float, float, float>(),
                                       rmsKernelCu<float>);

    // Matmul
    registerKernel<MatmulKernelFn<float, float, float, float>>(OpType::kMatmul,
                                                               eUTIL::DeviceType::kCpu,
                                                               makeTensorSignature<float, float, float, float>(),
                                                               matmulKernelCpu<float, float, float, float>);
    registerKernel<MatmulKernelFn<float, float, float, float>>(OpType::kMatmul,
                                                               eUTIL::DeviceType::kCuda,
                                                               makeTensorSignature<float, float, float, float>(),
                                                               matmulKernelCu<float, float, float, float>);
    registerKernel<MatmulKernelFn<float, int8_t, float, float>>(OpType::kMatmul,
                                                                eUTIL::DeviceType::kCuda,
                                                                makeTensorSignature<float, int8_t, float, float>(),
                                                                matmulKernelCu<float, int8_t, float, float>);

    // Swiglu
    registerKernel<SwigluKernelFn<float>>(OpType::kSwiglu,
                                          eUTIL::DeviceType::kCpu,
                                          makeTensorSignature<float, float, float>(),
                                          swigluKernelCpu<float>);
    registerKernel<SwigluKernelFn<float>>(OpType::kSwiglu,
                                          eUTIL::DeviceType::kCuda,
                                          makeTensorSignature<float, float, float>(),
                                          swigluKernelCu<float>);

    // Softmax
    registerKernel<SoftmaxKernelFn<float>>(OpType::kSoftmax,
                                           eUTIL::DeviceType::kCpu,
                                           makeTensorSignature<float>(),
                                           softmaxInplaceCpu<float>);
    
    // Scalesum
    registerKernel<ScalesumKernelFn<float>>(OpType::kScalesum,
                                            eUTIL::DeviceType::kCpu,
                                            makeTensorSignature<float, float, float>(),
                                            scalesumKernelCpu<float>);
    
    // Mha
    registerKernel<MhaKernelFn<float>>(OpType::kMha,
                                       eUTIL::DeviceType::kCpu,
                                       makeTensorSignature<float, float, float, float, float>(),
                                       mhaKernelCpu<float>);
    registerKernel<MhaKernelFn<float>>(OpType::kMha,
                                       eUTIL::DeviceType::kCuda,
                                       makeTensorSignature<float, float, float, float, float>(),
                                       mhaKernelCu<float>);                                                                                               
}

}  // namespace eCEL
