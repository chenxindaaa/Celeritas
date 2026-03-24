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
    class TensorHandleBase {
    public:
        virtual ~TensorHandleBase() = default;
    };

    template <typename T>
    class TensorHandle final : public TensorHandleBase {
    public:
        template <typename... Args>
        explicit TensorHandle(Args&&... args)
            : tensor(std::make_unique<eUTIL::Tensor<T>>(std::forward<Args>(args)...)) {}

        eUTIL::Tensor<T>& get() { return *tensor; }

    private:
        std::unique_ptr<eUTIL::Tensor<T>> tensor;
    };

    void readCheckPoint();
    virtual void createLayers() = 0;

    template <typename T, typename... Args>
    eUTIL::Tensor<T>& createTensor(Args&&... args) {
        auto tensor = std::make_unique<TensorHandle<T>>(std::forward<Args>(args)...);
        eUTIL::Tensor<T>& tensorRef = tensor->get();
        m_tensors.push_back(std::move(tensor));
        return tensorRef;
    }

    template <typename LayerT, typename... Args>
    LayerT& createLayer(Args&&... args) {
        static_assert(std::is_base_of_v<Layer, LayerT>,
                      "LayerT must derive from Layer");
        auto layer = std::make_unique<LayerT>(std::forward<Args>(args)...);
        LayerT& layerRef = *layer;
        m_layers.push_back(std::move(layer));
        return layerRef;
    }
    
protected:
    std::string m_checkPointPath;
    std::string m_tokenizerPath;
    ModelConfig m_config;
    eUTIL::DeviceType m_device;
    std::unique_ptr<model::RawModelData> m_rawData;
    std::vector<std::unique_ptr<Layer>> m_layers;
    std::vector<std::unique_ptr<TensorHandleBase>> m_tensors;
};
}  // namespace eCEL
