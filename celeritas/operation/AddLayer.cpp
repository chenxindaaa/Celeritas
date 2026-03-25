#include "AddLayer.h"

namespace eCEL {

template <typename T>
void AddLayer<T>::forward(const ForwardContext& ctx,
                          const Tensor& input,
                          Tensor& output) {
    auto kernel = KernelFactory::getAddKernel();
    kernel(input, m_params.value.view(), output, nullptr);

    (void)ctx;
}

template class AddLayer<float>;

}  // namespace eCEL
