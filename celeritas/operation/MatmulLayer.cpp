#include "MatmulLayer.h"

namespace eCEL {

template <typename Tact, typename Tweight>
void MatmulLayer<Tact, Tweight>::forward(const ForwardContext& ctx,
                                         const Tensor& input,
                                         Tensor& output) {
    auto kernel = KernelFactory::getMatmulKernel();
    kernel(input,
           m_params.weight.view(),
           m_params.scaler.view(),
           output,
           m_params.group_size,
           nullptr);

    (void)ctx;
}

template class MatmulLayer<float, float>;
template class MatmulLayer<float, int8_t>;

}  // namespace eCEL
