#include "KernelFactory.h"

namespace eCEL {
    
KernelRegistry& Dispatcher::m_kernelRegistry = KernelRegistry::getInstance();

AddDispatcher KernelFactory::getAddKernel() {
    return AddDispatcher(OpType::kAdd);
}

EmbDispatcher KernelFactory::getEmbKernel() {
    return EmbDispatcher(OpType::kEmb);
}

RmsDispatcher KernelFactory::getRmsKernel() {
    return RmsDispatcher(OpType::kRms);
}

MatmulDispatcher KernelFactory::getMatmulKernel() {
    return MatmulDispatcher(OpType::kMatmul);
}

}  // namespace eCEL
