#pragma once

#include "KernelDispatcher.h"

namespace eCEL {

class KernelFactory final {
public:
    KernelFactory() = delete;

    static AddDispatcher getAddKernel();
    static EmbDispatcher getEmbKernel();
    static RmsDispatcher getRmsKernel();
};

}  // namespace eCEL