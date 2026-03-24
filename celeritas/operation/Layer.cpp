#include "Layer.h"

namespace eCEL {

Layer::Layer(eUTIL::DeviceType device,
             LayerType type,
             std::size_t inputCount,
             std::size_t outputCount,
             std::size_t weightCount,
             std::string name)
    : m_device(device),
      m_type(type),
      m_name(std::move(name)),
      m_inputs(inputCount),
      m_outputs(outputCount),
      m_weights(weightCount) {}

void Layer::validateBindings() const {
    validateTensorList(m_inputs, "input");
    validateTensorList(m_outputs, "output");
    validateTensorList(m_weights, "weight");
}

eUTIL::DType Layer::inputDType(std::size_t index) const {
    if (index >= m_inputs.size() || m_inputs[index].tensor == nullptr) {
        throw std::invalid_argument("input tensor is not set");
    }
    return m_inputs[index].dtype;
}

eUTIL::DType Layer::outputDType(std::size_t index) const {
    if (index >= m_outputs.size() || m_outputs[index].tensor == nullptr) {
        throw std::invalid_argument("output tensor is not set");
    }
    return m_outputs[index].dtype;
}

eUTIL::DType Layer::weightDType(std::size_t index) const {
    if (index >= m_weights.size() || m_weights[index].tensor == nullptr) {
        throw std::invalid_argument("weight tensor is not set");
    }
    return m_weights[index].dtype;
}

void Layer::validateTensorList(const std::vector<ConstTensorBinding>& tensors, const char* label) {
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].tensor == nullptr) {
            throw std::invalid_argument(std::string(label) + " tensor is not set");
        }
    }
}

void Layer::validateTensorList(const std::vector<MutableTensorBinding>& tensors, const char* label) {
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].tensor == nullptr) {
            throw std::invalid_argument(std::string(label) + " tensor is not set");
        }
    }
}

AddLayer::AddLayer(eUTIL::DeviceType device, std::string name)
    : Layer(device, LayerType::kAdd, 2, 1, 0, std::move(name)) {}

Status AddLayer::forward() {
    try {
        validateBindings();
        const auto dtype = inputDType(0);
        if (dtype != inputDType(1) || dtype != outputDType(0)) {
            throw std::invalid_argument("add layer requires matching tensor dtypes");
        }

        auto kernel = KernelFactory::getAddKernel();
        switch (dtype) {
            case eUTIL::DType::kInt8:
                kernel(input<int8_t>(0), input<int8_t>(1), output<int8_t>(0), nullptr);
                return Status::kOk;
            case eUTIL::DType::kFloat16:
                kernel(input<eUTIL::float16>(0), input<eUTIL::float16>(1),
                       output<eUTIL::float16>(0), nullptr);
                return Status::kOk;
            case eUTIL::DType::kFloat32:
                kernel(input<float>(0), input<float>(1), output<float>(0), nullptr);
                return Status::kOk;
            default:
                throw std::invalid_argument("unsupported add tensor dtype");
        }
    } catch (const std::invalid_argument&) {
        return Status::kInvalidArgument;
    } catch (...) {
        return Status::kRuntimeError;
    }
}

EmbeddingLayer::EmbeddingLayer(eUTIL::DeviceType device,
                               int32_t vocabSize,
                               std::string name)
    : Layer(device, LayerType::kEmb, 1, 1, 1, std::move(name)),
      m_vocabSize(vocabSize) {}

Status EmbeddingLayer::forward() {
    try {
        validateBindings();
        const auto dtype = inputDType(0);
        if (dtype != weightDType(0) || dtype != outputDType(0)) {
            throw std::invalid_argument("embedding layer requires matching tensor dtypes");
        }

        auto kernel = KernelFactory::getEmbKernel();
        switch (dtype) {
            case eUTIL::DType::kInt8:
                kernel(input<int8_t>(0), weight<int8_t>(0), output<int8_t>(0), m_vocabSize, nullptr);
                return Status::kOk;
            case eUTIL::DType::kFloat16:
                kernel(input<eUTIL::float16>(0), weight<eUTIL::float16>(0),
                       output<eUTIL::float16>(0), m_vocabSize, nullptr);
                return Status::kOk;
            case eUTIL::DType::kFloat32:
                kernel(input<float>(0), weight<float>(0), output<float>(0), m_vocabSize, nullptr);
                return Status::kOk;
            default:
                throw std::invalid_argument("unsupported embedding tensor dtype");
        }
    } catch (const std::invalid_argument&) {
        return Status::kInvalidArgument;
    } catch (...) {
        return Status::kRuntimeError;
    }
}

RmsNormLayer::RmsNormLayer(eUTIL::DeviceType device, std::string name)
    : Layer(device, LayerType::kRms, 1, 1, 1, std::move(name)) {}

Status RmsNormLayer::forward() {
    try {
        validateBindings();
        const auto dtype = inputDType(0);
        if (dtype != weightDType(0) || dtype != outputDType(0)) {
            throw std::invalid_argument("rmsnorm layer requires matching tensor dtypes");
        }

        auto kernel = KernelFactory::getRmsKernel();
        switch (dtype) {
            case eUTIL::DType::kInt8:
                kernel(input<int8_t>(0), weight<int8_t>(0), output<int8_t>(0), nullptr);
                return Status::kOk;
            case eUTIL::DType::kFloat16:
                kernel(input<eUTIL::float16>(0), weight<eUTIL::float16>(0),
                       output<eUTIL::float16>(0), nullptr);
                return Status::kOk;
            case eUTIL::DType::kFloat32:
                kernel(input<float>(0), weight<float>(0), output<float>(0), nullptr);
                return Status::kOk;
            default:
                throw std::invalid_argument("unsupported rmsnorm tensor dtype");
        }
    } catch (const std::invalid_argument&) {
        return Status::kInvalidArgument;
    } catch (...) {
        return Status::kRuntimeError;
    }
}

MatmultLayer::MatmultLayer(eUTIL::DeviceType device,
                           float scale,
                           std::string name)
    : Layer(device, LayerType::kMatmul, 1, 1, 1, std::move(name)),
      m_scale(scale),
      m_groupSize(1),
      m_cudaConfig(nullptr) {}

Status MatmultLayer::forward() {
    try {
        validateBindings();

        eUTIL::Tensor<float> scaler(eUTIL::DeviceType::kCpu, 1);
        scaler[0] = m_scale;
        if (m_device == eUTIL::DeviceType::kCuda) {
            scaler.cuda();
        }

        const auto inputDtype = inputDType(0);
        const auto weightDtype = weightDType(0);
        const auto outputDtype = outputDType(0);
        auto kernel = KernelFactory::getMatmulKernel();

        if (inputDtype == eUTIL::DType::kFloat32 &&
            weightDtype == eUTIL::DType::kFloat32 &&
            outputDtype == eUTIL::DType::kFloat32) {
            kernel(input<float>(0), weight<float>(0), scaler, output<float>(0),
                   m_groupSize, m_cudaConfig);
            return Status::kOk;
        }

        if (inputDtype == eUTIL::DType::kFloat32 &&
            weightDtype == eUTIL::DType::kInt8 &&
            outputDtype == eUTIL::DType::kFloat32) {
            kernel(input<float>(0), weight<int8_t>(0), scaler, output<float>(0),
                   m_groupSize, m_cudaConfig);
            return Status::kOk;
        }

        throw std::invalid_argument("unsupported matmul tensor dtype combination");
    } catch (const std::invalid_argument&) {
        return Status::kInvalidArgument;
    } catch (...) {
        return Status::kRuntimeError;
    }
}

}  // namespace eCEL
