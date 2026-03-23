#include <algorithm>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>

#include "core/Tensor.h"
#include "memory/MemoryMgr.h"
#include "utils/utils.hpp"

namespace eUTIL {

namespace {

const char* deviceTypeName(DeviceType device) {
    switch (device) {
        case DeviceType::kCpu:
            return "cpu";
        case DeviceType::kCuda:
            return "cuda";
        case DeviceType::kUnknown:
        case DeviceType::kNumDeviceTypes:
        default:
            return "unknown";
    }
}

const char* dtypeName(DType dtype) {
    switch (dtype) {
        case DType::kInt32:
            return "int32";
        case DType::kFloat32:
            return "float32";
        case DType::kFloat64:
            return "float64";
        case DType::kUnknown:
        case DType::kNumDTypes:
        default:
            return "unknown";
    }
}

}  // namespace

template <typename T>
std::size_t Tensor<T>::calcSize(std::initializer_list<std::size_t> dims) {
    if (dims.size() == 0) {
        throw std::invalid_argument("Tensor dims cannot be empty");
    }

    for (const std::size_t dim : dims) {
        if (dim == 0) {
            throw std::invalid_argument("Tensor dim must be greater than 0");
        }
    }

    return std::accumulate(dims.begin(), dims.end(),
                           static_cast<std::size_t>(1), std::multiplies<std::size_t>());
}

template <typename T>
Tensor<T>::Tensor(DeviceType device, std::initializer_list<std::size_t> dims)
    : m_size(calcSize(dims)),
      m_dims(dims),
      m_device(device),
      m_dtype(DTypeTrait<T>::kValue),
      m_data(nullptr),
      m_control(nullptr) {
    if (m_device == DeviceType::kUnknown) {
        throw std::invalid_argument("Tensor device cannot be kUnknown");
    }

    auto& mgr = MemoryMgr::getInstance();
    switch (m_device) {
        case DeviceType::kCpu:
            m_data = mgr.allocateTyped<T>(DeviceType::kCpu, m_size);
            break;
        case DeviceType::kCuda:
            m_data = mgr.allocateTyped<T>(DeviceType::kCuda, m_size);
            break;
        case DeviceType::kUnknown:
        case DeviceType::kNumDeviceTypes:
        default:
            throw std::runtime_error("Unsupported device type");
    }

    m_control = new ControlBlock{1, false};
}

template <typename T>
Tensor<T>::Tensor() noexcept
    : m_size(0),
      m_dims(),
      m_device(DeviceType::kUnknown),
      m_dtype(DType::kUnknown),
      m_data(nullptr),
      m_control(nullptr) {}

template <typename T>
Tensor<T>::Tensor(DeviceType device,
                  std::initializer_list<std::size_t> dims,
                  T* data,
                  bool isExternal)
    : m_size(calcSize(dims)),
      m_dims(dims),
      m_device(device),
      m_dtype(DTypeTrait<T>::kValue),
      m_data(nullptr),
      m_control(nullptr) {
    if (m_device == DeviceType::kUnknown) {
        throw std::invalid_argument("Tensor device cannot be kUnknown");
    }
    if (data == nullptr) {
        throw std::invalid_argument("Tensor external data cannot be null");
    }
    if (isExternal) {
        m_data = data;
    }
    else {
        auto& mgr = MemoryMgr::getInstance();
        switch (m_device) {
            case DeviceType::kCpu:
                m_data = mgr.allocateTyped<T>(DeviceType::kCpu, m_size);
                break;
            case DeviceType::kCuda:
                m_data = mgr.allocateTyped<T>(DeviceType::kCuda, m_size);
                break;
            case DeviceType::kUnknown:
            case DeviceType::kNumDeviceTypes:
            default:
                throw std::runtime_error("Unsupported device type");
        }

        try {
            mgr.memcpy(m_device, m_data, m_device, data, sizeof(T) * m_size);
        } catch (...) {
            mgr.releaseTyped<T>(m_device, m_data, m_size);
            m_data = nullptr;
            throw;
        }
    }
    m_control = new ControlBlock{1, isExternal};
}

template <typename T>
Tensor<T>::~Tensor() noexcept {
    releaseOwnership();
}

template <typename T>
void Tensor<T>::acquireFrom(const Tensor& other) {
    m_size = other.m_size;
    m_dims = other.m_dims;
    m_device = other.m_device;
    m_dtype = other.m_dtype;
    m_data = other.m_data;
    m_control = other.m_control;
    if (m_control) {
        ++m_control->refCount;
    }
}

template <typename T>
void Tensor<T>::releaseOwnership() noexcept {
    if (!m_control) {
        return;
    }

    if (--m_control->refCount == 0) {
        release();
        delete m_control;
    }

    m_control = nullptr;
    m_data = nullptr;
    m_size = 0;
    m_dims.clear();
    m_device = DeviceType::kUnknown;
    m_dtype = DType::kUnknown;
}

template <typename T>
Tensor<T>::Tensor(const Tensor& other)
    : m_size(0),
      m_dims(),
      m_device(DeviceType::kUnknown),
      m_dtype(DType::kUnknown),
      m_data(nullptr),
      m_control(nullptr) {
    acquireFrom(other);
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }

    // Acquire first for strong exception safety.
    Tensor temp(other);
    releaseOwnership();
    acquireFrom(temp);
    return *this;
}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : m_size(other.m_size),
      m_dims(std::move(other.m_dims)),
      m_device(other.m_device),
      m_dtype(other.m_dtype),
      m_data(other.m_data),
      m_control(other.m_control) {
    other.m_size = 0;
    other.m_dims.clear();
    other.m_device = DeviceType::kUnknown;
    other.m_dtype = DType::kUnknown;
    other.m_data = nullptr;
    other.m_control = nullptr;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    releaseOwnership();
    m_size = other.m_size;
    m_dims = std::move(other.m_dims);
    m_device = other.m_device;
    m_dtype = other.m_dtype;
    m_data = other.m_data;
    m_control = other.m_control;

    other.m_size = 0;
    other.m_dims.clear();
    other.m_device = DeviceType::kUnknown;
    other.m_dtype = DType::kUnknown;
    other.m_data = nullptr;
    other.m_control = nullptr;
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::clone() const {
    if (m_device == DeviceType::kUnknown) {
        throw std::runtime_error("Cannot clone tensor with kUnknown device");
    }

    Tensor copied;
    copied.m_size = m_size;
    copied.m_dims = m_dims;
    copied.m_device = m_device;
    copied.m_dtype = m_dtype;

    auto& mgr = MemoryMgr::getInstance();
    switch (copied.m_device) {
        case DeviceType::kCpu:
            copied.m_data = mgr.allocateTyped<T>(DeviceType::kCpu, copied.m_size);
            break;
        case DeviceType::kCuda:
            copied.m_data = mgr.allocateTyped<T>(DeviceType::kCuda, copied.m_size);
            break;
        case DeviceType::kUnknown:
        case DeviceType::kNumDeviceTypes:
        default:
            throw std::runtime_error("Unsupported device type");
    }
    copied.m_control = new ControlBlock{1, false};

    if (m_data != nullptr && m_size != 0) {
        mgr.memcpy(m_device, copied.m_data, m_device, m_data, sizeof(T) * m_size);
    }
    return copied;
}

template <typename T>
Tensor<T>& Tensor<T>::cpu() {
    if (m_device == DeviceType::kUnknown) {
        throw std::runtime_error("Cannot convert tensor with kUnknown device");
    }

    if (m_device == DeviceType::kCpu) {
        return *this;
    }
    if (m_data == nullptr || m_size == 0) {
        m_device = DeviceType::kCpu;
        return *this;
    }

    auto& mgr = MemoryMgr::getInstance();
    T* hostData = mgr.allocateTyped<T>(DeviceType::kCpu, m_size);
    try {
        mgr.memcpy(DeviceType::kCpu, hostData, m_device, m_data, sizeof(T) * m_size);
    } catch (...) {
        mgr.releaseTyped<T>(DeviceType::kCpu, hostData, m_size);
        throw;
    }

    const std::size_t originalSize = m_size;
    const std::vector<std::size_t> originalDims = m_dims;
    const DType originalDType = m_dtype;
    releaseOwnership();
    m_data = hostData;
    m_size = originalSize;
    m_dims = originalDims;
    m_device = DeviceType::kCpu;
    m_dtype = originalDType;
    m_control = new ControlBlock{1, false};
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::cuda(cudaStream_t stream) {
    if (m_device == DeviceType::kUnknown) {
        throw std::runtime_error("Cannot convert tensor with kUnknown device");
    }

    if (m_device == DeviceType::kCuda) {
        return *this;
    }
    if (m_data == nullptr || m_size == 0) {
        m_device = DeviceType::kCuda;
        return *this;
    }

    auto& mgr = MemoryMgr::getInstance();
    T* deviceData = mgr.allocateTyped<T>(DeviceType::kCuda, m_size);
    try {
        mgr.memcpy(DeviceType::kCuda, deviceData, m_device, m_data, sizeof(T) * m_size, stream);
    } catch (...) {
        mgr.releaseTyped<T>(DeviceType::kCuda, deviceData, m_size);
        throw;
    }

    const std::size_t originalSize = m_size;
    const std::vector<std::size_t> originalDims = m_dims;
    const DType originalDType = m_dtype;
    releaseOwnership();
    m_data = deviceData;
    m_size = originalSize;
    m_dims = originalDims;
    m_device = DeviceType::kCuda;
    m_dtype = originalDType;
    m_control = new ControlBlock{1, false};
    return *this;
}

template <typename T>
void Tensor<T>::release() noexcept {
    if (m_data == nullptr || m_control == nullptr || m_control->isExternal) {
        return;
    }

    try {
        auto& mgr = MemoryMgr::getInstance();
        if (m_device == DeviceType::kCpu) {
            mgr.releaseTyped<T>(DeviceType::kCpu, m_data, m_size);
        } else if (m_device == DeviceType::kCuda) {
            mgr.releaseTyped<T>(DeviceType::kCuda, m_data, m_size);
        } else {
            // kUnknown should never own memory.
        }
    } catch (...) {
        // Destructors and deleters must not throw.
    }
}

template <typename T>
std::string Tensor<T>::toString() const {
    std::ostringstream os;
    os << "Tensor(shape=[";
    for (std::size_t i = 0; i < m_dims.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << m_dims[i];
    }
    os << "], device=" << deviceTypeName(m_device)
       << ", dtype=" << dtypeName(m_dtype)
       << ", size=" << m_size
       << ", data=";

    if (empty()) {
        os << "[]";
        os << ")";
        return os.str();
    }

    constexpr std::size_t kPreviewCount = 16;
    if (m_device == DeviceType::kUnknown) {
        os << "<unavailable>";
        os << ")";
        return os.str();
    }

    os << "[";
    const T* previewData = data();
    std::unique_ptr<Tensor<T>> hostPreview;
    if (m_device == DeviceType::kCuda) {
        hostPreview = std::make_unique<Tensor<T>>(DeviceType::kCpu, m_size);
        MemoryMgr::getInstance().memcpy(DeviceType::kCpu, hostPreview->data(), m_device,
                                        m_data, byteSize());
        previewData = hostPreview->data();
    }

    const std::size_t previewCount = std::min(kPreviewCount, m_size);
    for (std::size_t i = 0; i < previewCount; ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << previewData[i];
    }
    if (m_size > kPreviewCount) {
        os << ", ...";
    }
    os << "])";
    return os.str();
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
    os << tensor.toString();
    return os;
}

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;
template std::ostream& operator<<(std::ostream& os, const Tensor<int>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<float>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<double>& tensor);

}  // namespace eUTIL
