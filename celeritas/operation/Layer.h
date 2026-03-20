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

template <typename T>
class Layer {
public:
    Layer(eUTIL::DeviceType device,
          LayerType type,
          std::size_t inputCount,
          std::size_t outputCount,
          std::size_t weightCount = 0,
          std::string name = "")
        : m_device(device),
          m_type(type),
          m_name(std::move(name)),
          m_inputs(inputCount, nullptr),
          m_outputs(outputCount, nullptr),
          m_weights(weightCount, nullptr) {}

    virtual ~Layer() = default;

    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    virtual Status forward() = 0;

    eUTIL::DeviceType device() const { return m_device; }
    LayerType type() const { return m_type; }
    const std::string& name() const { return m_name; }
    void setName(const std::string& name) { m_name = name; }

    void setInput(std::size_t index, const eUTIL::Tensor<T>& tensor) {
        if (index >= m_inputs.size()) {
            throw std::out_of_range("input index out of range");
        }
        m_inputs[index] = &tensor;
    }

    void setOutput(std::size_t index, eUTIL::Tensor<T>& tensor) {
        if (index >= m_outputs.size()) {
            throw std::out_of_range("output index out of range");
        }
        m_outputs[index] = &tensor;
    }

    void setWeight(std::size_t index, const eUTIL::Tensor<T>& tensor) {
        if (index >= m_weights.size()) {
            throw std::out_of_range("weight index out of range");
        }
        m_weights[index] = &tensor;
    }

    const eUTIL::Tensor<T>& input(std::size_t index) const {
        if (index >= m_inputs.size() || m_inputs[index] == nullptr) {
            throw std::invalid_argument("input tensor is not set");
        }
        return *m_inputs[index];
    }

    eUTIL::Tensor<T>& output(std::size_t index) const {
        if (index >= m_outputs.size() || m_outputs[index] == nullptr) {
            throw std::invalid_argument("output tensor is not set");
        }
        return *m_outputs[index];
    }

    const eUTIL::Tensor<T>& weight(std::size_t index) const {
        if (index >= m_weights.size() || m_weights[index] == nullptr) {
            throw std::invalid_argument("weight tensor is not set");
        }
        return *m_weights[index];
    }

    std::size_t inputCount() const { return m_inputs.size(); }
    std::size_t outputCount() const { return m_outputs.size(); }
    std::size_t weightCount() const { return m_weights.size(); }
    bool hasWeight() const { return !m_weights.empty(); }

protected:
    void validateBindings() const {
        validateTensorList(m_inputs, "input");
        validateTensorList(m_outputs, "output");
        validateTensorList(m_weights, "weight");
    }

private:
    template <typename TensorPtr>
    static void validateTensorList(const std::vector<TensorPtr>& tensors, const char* label) {
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            if (tensors[i] == nullptr) {
                throw std::invalid_argument(std::string(label) + " tensor is not set");
            }
        }
    }

protected:
    eUTIL::DeviceType m_device;
    LayerType m_type;
    std::string m_name;
    std::vector<const eUTIL::Tensor<T>*> m_inputs;
    std::vector<eUTIL::Tensor<T>*> m_outputs;
    std::vector<const eUTIL::Tensor<T>*> m_weights;
};

template <typename T>
class AddLayer final : public Layer<T> {
public:
    explicit AddLayer(eUTIL::DeviceType device, std::string name = "AddLayer")
        : Layer<T>(device, LayerType::kAdd, 2, 1, 0, std::move(name)) {}

    Status forward() override {
        try {
            this->validateBindings();
            auto kernel = KernelFactory::getAddKernel();
            kernel(this->input(0), this->input(1), this->output(0), nullptr);
            return Status::kOk;
        } catch (const std::invalid_argument&) {
            return Status::kInvalidArgument;
        } catch (...) {
            return Status::kRuntimeError;
        }
    }
};

template <typename T>
class EmbeddingLayer final : public Layer<T> {
public:
    explicit EmbeddingLayer(eUTIL::DeviceType device,
                            int32_t vocabSize,
                            std::string name = "EmbeddingLayer")
        : Layer<T>(device, LayerType::kEmb, 1, 1, 1, std::move(name)),
          m_vocabSize(vocabSize) {}

    void setVocabSize(int32_t vocabSize) { m_vocabSize = vocabSize; }
    int32_t vocabSize() const { return m_vocabSize; }

    Status forward() override {
        try {
            this->validateBindings();
            auto kernel = KernelFactory::getEmbKernel();
            kernel(this->input(0), this->weight(0), this->output(0), m_vocabSize, nullptr);
            return Status::kOk;
        } catch (const std::invalid_argument&) {
            return Status::kInvalidArgument;
        } catch (...) {
            return Status::kRuntimeError;
        }
    }

private:
    int32_t m_vocabSize;
};

template <typename T>
class RmsNormLayer final : public Layer<T> {
public:
    explicit RmsNormLayer(eUTIL::DeviceType device, std::string name = "RmsNormLayer")
        : Layer<T>(device, LayerType::kRms, 1, 1, 1, std::move(name)) {}

    Status forward() override {
        try {
            this->validateBindings();
            auto kernel = KernelFactory::getRmsKernel();
            kernel(this->input(0), this->weight(0), this->output(0), nullptr);
            return Status::kOk;
        } catch (const std::invalid_argument&) {
            return Status::kInvalidArgument;
        } catch (...) {
            return Status::kRuntimeError;
        }
    }
};

template <typename T>
class MatmultLayer final : public Layer<T> {
public:
    explicit MatmultLayer(eUTIL::DeviceType device,
                          float scale = 1.f,
                          std::string name = "MatmultLayer")
        : Layer<T>(device, LayerType::kMatmul, 1, 1, 1, std::move(name)),
          m_scale(scale),
          m_cudaConfig(nullptr) {}

    void setScale(float scale) { m_scale = scale; }
    float scale() const { return m_scale; }

    void setCudaConfig(const eUTIL::CudaConfig* config) { m_cudaConfig = config; }
    const eUTIL::CudaConfig* cudaConfig() const { return m_cudaConfig; }

    Status forward() override {
        try {
            this->validateBindings();
            auto kernel = KernelFactory::getMatmulKernel();
            kernel(this->input(0), this->weight(0), this->output(0), m_scale, m_cudaConfig);
            return Status::kOk;
        } catch (const std::invalid_argument&) {
            return Status::kInvalidArgument;
        } catch (...) {
            return Status::kRuntimeError;
        }
    }

private:
    float m_scale;
    const eUTIL::CudaConfig* m_cudaConfig;
};

}  // namespace eCEL
