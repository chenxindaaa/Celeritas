#include "SwigluLayer.h"

namespace eCEL {

template <typename T>
void SwigluLayer<T>::forward(const ForwardContext& ctx,
                             const Tensor& input,
                             Tensor& output) {
    auto kernel = KernelFactory::getSwigluKernel();
    kernel(input, m_params.value.view(), output, nullptr);

    (void)ctx;
}

template class SwigluLayer<float>;

}  // namespace eCEL
