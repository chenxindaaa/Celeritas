#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "./config.h"
#include "./RawData.h"
#include "celeritas/operation/Layer.h"
#include "udm/common/BaseTypes.h"

namespace eCEL {
class Model {
public:
    virtual ~Model() = default;
    Model(std::string checkPointPath, std::string tokenizerPath, 
          eUTIL::DeviceType device = eUTIL::DeviceType::kCpu) :
          m_checkPointPath(std::move(checkPointPath)),
          m_tokenizerPath(std::move(tokenizerPath)),
          m_device(device) {}
    
    void init();

protected:
    void readCheckPoint();
    virtual void createLayers() = 0;
    
protected:
    std::string m_checkPointPath;
    std::string m_tokenizerPath;
    TransformerConfig m_config;
    eUTIL::DeviceType m_device;
    std::unique_ptr<model::RawModelData> m_rawData;
};
}  // namespace eCEL
