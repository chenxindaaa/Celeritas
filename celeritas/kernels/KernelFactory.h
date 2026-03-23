#pragma once

#include "KernelDispatcher.h"

namespace eCEL {

class KernelFactory final {
public:
    KernelFactory() = delete;

    static AddDispatcher getAddKernel();
    static EmbDispatcher getEmbKernel();
    static RmsDispatcher getRmsKernel();
    static MatmulDispatcher getMatmulKernel();
    static SwigluDispatcher getSwigluKernel();
    static SoftmaxDispatcher getSoftmaxKernel();
    static ScalesumDispatcher getScalesumKernel();
    static MhaDispatcher getMhaKernel();
};

}  // namespace eCEL