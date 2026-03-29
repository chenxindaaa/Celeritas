#pragma once

// Defines the common model base and the loaded state shared by concrete models.

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "./RawData.h"
#include "./config.h"
#include "celeritas/operation/Layer.h"
#include "celeritas/tokenizer/Tokenizer.h"
#include "udm/common/BaseTypes.h"

namespace eCEL {

class Loader;

class Model {
public:
    virtual ~Model() = default;

    explicit Model(std::string checkpointPath,
                   eUTIL::DeviceType device = eUTIL::DeviceType::kCpu)
        : m_checkpointPath(std::move(checkpointPath)),
          m_device(device) {}

    virtual std::vector<float> forward(const ModelInputs& inputs) const;
    virtual std::vector<float> forward(const ModelInputs& inputs,
                                       const ForwardContext& ctx) const;

    void to(eUTIL::DeviceType device);
    void cpu() { to(eUTIL::DeviceType::kCpu); }
    void cuda() { to(eUTIL::DeviceType::kCuda); }

    eUTIL::DeviceType device() const { return m_device; }
    const TransformerConfig& config() const { return m_config; }
    const std::string& checkpointPath() const { return m_checkpointPath; }
    bool isLoaded() const { return m_isLoaded; }

protected:
    virtual void createLayers() = 0;
    virtual void moveToDevice(eUTIL::DeviceType device) = 0;
    virtual void loadWeights() = 0;

protected:
    std::string m_checkpointPath;
    TransformerConfig m_config;
    eUTIL::DeviceType m_device;
    std::unique_ptr<RawModelData> m_rawData;
    bool m_isLoaded = false;

private:
    friend class Loader;

    // Stores loaded checkpoint state before model-specific layer creation.
    void setLoadedState(TransformerConfig config,
                        std::unique_ptr<RawModelData> rawData) {
        if (rawData == nullptr) {
            throw std::invalid_argument("Loaded raw model data must not be null");
        }

        m_config = config;
        m_rawData = std::move(rawData);
        m_isLoaded = true;
    }
};

}  // namespace eCEL
