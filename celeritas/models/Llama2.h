#pragma once

#include <memory>
#include <vector>

#include "Model.h"
#include "../operation/DecoderLayer.h"
#include "../operation/EmbLayer.h"
#include "../operation/Layer.h"
#include "../operation/MatmulLayer.h"
#include "../operation/RmsnormLayer.h"
#include "udm/core/Tensor.h"

namespace eCEL {
class Llama2 : public Model
{
public:
    virtual ~Llama2() override = default;
    Llama2(std::string checkpointPath,
          eUTIL::DeviceType device = eUTIL::DeviceType::kCpu):
          Model(checkpointPath, device) {}
    std::vector<float> forward(const ModelInputs& inputs) const override;
    std::vector<float> forward(const ModelInputs& inputs,
                               const ForwardContext& ctx) const override;
protected:
    void createLayers() override;
    void moveToDevice(eUTIL::DeviceType device) override;
    void loadWeights() override;

private:
    void validateConfig() const;
    void loadLegacyWeights();
    void loadVersion1Weights();

    std::vector<std::unique_ptr<DecoderLayer<float, float>>> m_decoderLayers;
    std::unique_ptr<EmbLayer<float>> m_embLayer;
    std::unique_ptr<RmsnormLayer<float>> m_finalNormLayer;
    std::unique_ptr<MatmulLayer<float, float>> m_lmHeadLayer;
};
}  // namespace eCEL
