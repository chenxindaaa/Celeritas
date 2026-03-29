#include "Model.h"

namespace eCEL {

std::vector<float> Model::forward(const ModelInputs& inputs) const {
    ForwardContext ctx;
    return forward(inputs, ctx);
}

std::vector<float> Model::forward(const ModelInputs& inputs,
                                  const ForwardContext& ctx) const {
    (void)inputs;
    (void)ctx;
    throw std::runtime_error("Model::forward(ModelInputs, ForwardContext) is not implemented for this model");
}

void Model::to(eUTIL::DeviceType device) {
    if (!m_isLoaded) {
        throw std::runtime_error("Model::to requires a loaded model");
    }
    moveToDevice(device);
    m_device = device;
}

}  // namespace eCEL
