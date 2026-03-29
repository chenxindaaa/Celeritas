#include "MatmulLayer.h"

namespace eCEL {

template <typename Tact, typename Tweight>
void MatmulLayer<Tact, Tweight>::forward(const ForwardContext& ctx,
                                         TensorListView<Tact> inputs,
                                         eUTIL::Tensor<Tact>& output) {
    Layer<Tact>::requireInputCount(inputs, 1, "MatmulLayer");

    auto kernel = KernelFactory::getMatmulKernel();
    kernel(inputs[0],
           m_params.weight.view(),
           m_params.scaler.view(),
           output,
           m_groupSize,
           nullptr);

    (void)ctx;
}

template class MatmulLayer<float, float>;
template class MatmulLayer<float, int8_t>;

}  // namespace eCEL
