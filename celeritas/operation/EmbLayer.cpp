#include "EmbLayer.h"

namespace eCEL {

template <typename T>
void EmbLayer<T>::forward(const ForwardContext& ctx,
                          TensorListView<T> inputs,
                          eUTIL::Tensor<T>& output) {
    Layer<T>::requireInputCount(inputs, 1, "EmbLayer");

    auto kernel = KernelFactory::getEmbKernel();
    kernel(inputs[0], m_params.weight.view(), output, m_vocabSize, nullptr);

    (void)ctx;
}

template class EmbLayer<float>;

}  // namespace eCEL
