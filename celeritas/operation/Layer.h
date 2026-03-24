#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "celeritas/kernels/KernelFactory.h"
#include "celeritas/kernels/KernelRegistry.h"
#include "udm/core/Tensor.h"

namespace eCEL {

enum class Status {
    kOk = 0,
    kInvalidArgument = 1,
    kRuntimeError = 2,
};

using LayerType = OpType;

class Layer {
public:
    Layer(eUTIL::DeviceType device,
          LayerType type,
          std::size_t inputCount,
          std::size_t outputCount,
          std::size_t weightCount = 0,
          std::string name = "");
    virtual ~Layer() = default;

    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    virtual Status forward() = 0;

    eUTIL::DeviceType device() const { return m_device; }
    LayerType type() const { return m_type; }
    const std::string& name() const { return m_name; }
    void setName(const std::string& name) { m_name = name; }

    std::size_t inputCount() const { return m_inputs.size(); }
    std::size_t outputCount() const { return m_outputs.size(); }
    std::size_t weightCount() const { return m_weights.size(); }
    bool hasWeight() const { return !m_weights.empty(); }

    template <typename T>
    void setInput(std::size_t index, const eUTIL::Tensor<T>& tensor) {
        if (index >= m_inputs.size()) {
            throw std::out_of_range("input index out of range");
        }
        m_inputs[index] = makeConstBinding(tensor);
    }

    template <typename T>
    void setOutput(std::size_t index, eUTIL::Tensor<T>& tensor) {
        if (index >= m_outputs.size()) {
            throw std::out_of_range("output index out of range");
        }
        m_outputs[index] = makeMutableBinding(tensor);
    }

    template <typename T>
    void setWeight(std::size_t index, const eUTIL::Tensor<T>& tensor) {
        if (index >= m_weights.size()) {
            throw std::out_of_range("weight index out of range");
        }
        m_weights[index] = makeConstBinding(tensor);
    }

protected:
    struct ConstTensorBinding {
        const void* tensor = nullptr;
        eUTIL::DType dtype = eUTIL::DType::kUnknown;
        eUTIL::DeviceType device = eUTIL::DeviceType::kUnknown;
    };

    struct MutableTensorBinding {
        void* tensor = nullptr;
        eUTIL::DType dtype = eUTIL::DType::kUnknown;
        eUTIL::DeviceType device = eUTIL::DeviceType::kUnknown;
    };

    void validateBindings() const;

    eUTIL::DType inputDType(std::size_t index) const;
    eUTIL::DType outputDType(std::size_t index) const;
    eUTIL::DType weightDType(std::size_t index) const;

    template <typename T>
    const eUTIL::Tensor<T>& input(std::size_t index) const {
        return accessConstTensor<T>(m_inputs, index, "input");
    }

    template <typename T>
    eUTIL::Tensor<T>& output(std::size_t index) const {
        return accessMutableTensor<T>(m_outputs, index, "output");
    }

    template <typename T>
    const eUTIL::Tensor<T>& weight(std::size_t index) const {
        return accessConstTensor<T>(m_weights, index, "weight");
    }

protected:
    eUTIL::DeviceType m_device;
    LayerType m_type;
    std::string m_name;
    std::vector<ConstTensorBinding> m_inputs;
    std::vector<MutableTensorBinding> m_outputs;
    std::vector<ConstTensorBinding> m_weights;

private:
    static void validateTensorList(const std::vector<ConstTensorBinding>& tensors, const char* label);
    static void validateTensorList(const std::vector<MutableTensorBinding>& tensors, const char* label);

    template <typename T>
    static ConstTensorBinding makeConstBinding(const eUTIL::Tensor<T>& tensor) {
        return ConstTensorBinding{&tensor, tensor.dtype(), tensor.device()};
    }

    template <typename T>
    static MutableTensorBinding makeMutableBinding(eUTIL::Tensor<T>& tensor) {
        return MutableTensorBinding{&tensor, tensor.dtype(), tensor.device()};
    }

    template <typename T>
    static const eUTIL::Tensor<T>& accessConstTensor(const std::vector<ConstTensorBinding>& tensors,
                                                     std::size_t index,
                                                     const char* label) {
        if (index >= tensors.size() || tensors[index].tensor == nullptr) {
            throw std::invalid_argument(std::string(label) + " tensor is not set");
        }
        if (tensors[index].dtype != eUTIL::DTypeTrait<T>::kValue) {
            throw std::invalid_argument(std::string(label) + " tensor dtype mismatch");
        }
        return *static_cast<const eUTIL::Tensor<T>*>(tensors[index].tensor);
    }

    template <typename T>
    static eUTIL::Tensor<T>& accessMutableTensor(const std::vector<MutableTensorBinding>& tensors,
                                                 std::size_t index,
                                                 const char* label) {
        if (index >= tensors.size() || tensors[index].tensor == nullptr) {
            throw std::invalid_argument(std::string(label) + " tensor is not set");
        }
        if (tensors[index].dtype != eUTIL::DTypeTrait<T>::kValue) {
            throw std::invalid_argument(std::string(label) + " tensor dtype mismatch");
        }
        return *static_cast<eUTIL::Tensor<T>*>(tensors[index].tensor);
    }
};

class AddLayer final : public Layer {
public:
    explicit AddLayer(eUTIL::DeviceType device, std::string name = "AddLayer");
    Status forward() override;
};

class EmbeddingLayer final : public Layer {
public:
    explicit EmbeddingLayer(eUTIL::DeviceType device,
                            int32_t vocabSize,
                            std::string name = "EmbeddingLayer");

    void setVocabSize(int32_t vocabSize) { m_vocabSize = vocabSize; }
    int32_t vocabSize() const { return m_vocabSize; }

    Status forward() override;

private:
    int32_t m_vocabSize;
};

class RmsNormLayer final : public Layer {
public:
    explicit RmsNormLayer(eUTIL::DeviceType device, std::string name = "RmsNormLayer");
    Status forward() override;
};

class MatmultLayer final : public Layer {
public:
    explicit MatmultLayer(eUTIL::DeviceType device,
                          float scale = 1.f,
                          std::string name = "MatmultLayer");

    void setScale(float scale) { m_scale = scale; }
    float scale() const { return m_scale; }
    void setGroupSize(int32_t groupSize) { m_groupSize = groupSize; }
    int32_t groupSize() const { return m_groupSize; }

    void setCudaConfig(const eUTIL::CudaConfig* config) { m_cudaConfig = config; }
    const eUTIL::CudaConfig* cudaConfig() const { return m_cudaConfig; }

    Status forward() override;

private:
    float m_scale;
    int32_t m_groupSize;
    const eUTIL::CudaConfig* m_cudaConfig;
};

}  // namespace eCEL
