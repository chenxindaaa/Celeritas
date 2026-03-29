#include "AddLayer.h"

namespace eCEL {

template <typename T>
void AddLayer<T>::forward(const ForwardContext& ctx,
                          TensorListView<T> inputs,
                          eUTIL::Tensor<T>& output) {
    Layer<T>::requireInputCount(inputs, 2, "AddLayer");

    auto kernel = KernelFactory::getAddKernel();
    kernel(inputs[0], inputs[1], output, nullptr);

    (void)ctx;
}

template class AddLayer<float>;

}  // namespace eCEL
