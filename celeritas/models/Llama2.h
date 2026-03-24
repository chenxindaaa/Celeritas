#pragma once

#include <memory>
#include <vector>

#include "Model.h"
#include "../operation/Layer.h"
#include "udm/core/Tensor.h"

namespace eCEL {
class Llama2 : public Model
{
public:
    virtual ~Llama2() override = default;
    Llama2(std::string checkPointPath, std::string tokenizerPath, 
          eUTIL::DeviceType device = eUTIL::DeviceType::kCpu):
          Model(checkPointPath, tokenizerPath, device) {}
protected:
    void createLayers() override;
    void createEmb();
};
}  // namespace eCEL
