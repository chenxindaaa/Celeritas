#include "SwigluLayer.h"

namespace eCEL {

template <typename T>
void SwigluLayer<T>::forward(const ForwardContext& ctx,
                             TensorListView<T> inputs,
                             eUTIL::Tensor<T>& output) {
    Layer<T>::requireInputCount(inputs, 2, "SwigluLayer");

    auto kernel = KernelFactory::getSwigluKernel();
    kernel(inputs[0], inputs[1], output, nullptr);

    (void)ctx;
}

template class SwigluLayer<float>;

}  // namespace eCEL
