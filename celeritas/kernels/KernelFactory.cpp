#include "KernelFactory.h"

namespace eCEL {

AddDispatcher KernelFactory::getAddKernel() {
    return AddDispatcher(OpType::kAdd);
}

EmbDispatcher KernelFactory::getEmbKernel() {
    return EmbDispatcher(OpType::kEmb);
}

}  // namespace eCEL
