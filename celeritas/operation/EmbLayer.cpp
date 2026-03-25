#include "EmbLayer.h"

namespace eCEL {

template <typename Tact, typename Tweight>
void EmbLayer<Tact, Tweight>::forward(const ForwardContext& ctx,
                                      const Tensor& input,
                                      Tensor& output) {
    auto kernel = KernelFactory::getEmbKernel();
    kernel(input, m_params.weight.view(), output, m_params.vocab_size, nullptr);

    (void)ctx;
}

template class EmbLayer<float, float>;

}  // namespace eCEL
