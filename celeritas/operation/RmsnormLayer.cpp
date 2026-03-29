#include "RmsnormLayer.h"

namespace eCEL {

template <typename T>
void RmsnormLayer<T>::forward(const ForwardContext& ctx,
                              TensorListView<T> inputs,
                              eUTIL::Tensor<T>& output) {
    Layer<T>::requireInputCount(inputs, 1, "RmsnormLayer");

    auto kernel = KernelFactory::getRmsKernel();
    kernel(inputs[0], m_params.weight.view(), output, nullptr);

    (void)ctx;
}

template class RmsnormLayer<float>;

}  // namespace eCEL
