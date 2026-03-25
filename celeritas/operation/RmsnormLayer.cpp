#include "RmsnormLayer.h"

namespace eCEL {

template <typename Tact, typename Tweight>
void RmsnormLayer<Tact, Tweight>::forward(const ForwardContext& ctx,
                                          const Tensor& input,
                                          Tensor& output) {
    auto kernel = KernelFactory::getRmsKernel();
    kernel(input, m_params.weight.view(), output, nullptr);

    (void)ctx;
}

template class RmsnormLayer<float, float>;

}  // namespace eCEL
