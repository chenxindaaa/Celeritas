#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>

#include "core/Tensor.h"
#include "memory/MemoryMgr.h"
#include "utils/utils.hpp"

namespace eUTIL {

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
      m_refCount(nullptr) {
    if (m_device == DeviceType::kUnknown) {
        throw std::invalid_argument("Tensor device cannot be kUnknown");
    }

    auto& mgr = MemoryMgr::getInstance();
    switch (m_device) {
        case DeviceType::kCpu:
            m_data = mgr.allocateTyped<T>(DeviceType::kCpu, nullptr, m_size);
            break;
        case DeviceType::kCuda:
            m_data = mgr.allocateTyped<T>(DeviceType::kCuda, nullptr, m_size);
            break;
        case DeviceType::kUnknown:
        case DeviceType::kNumDeviceTypes:
        default:
            throw std::runtime_error("Unsupported device type");
    }

    m_refCount = new std::size_t(1);
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
    m_refCount = other.m_refCount;
    if (m_refCount) {
        ++(*m_refCount);
    }
}

template <typename T>
void Tensor<T>::releaseOwnership() noexcept {
    if (!m_refCount) {
        return;
    }

    if (--(*m_refCount) == 0) {
        release();
        delete m_refCount;
    }

    m_refCount = nullptr;
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
      m_refCount(nullptr) {
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
      m_refCount(other.m_refCount) {
    other.m_size = 0;
    other.m_dims.clear();
    other.m_device = DeviceType::kUnknown;
    other.m_dtype = DType::kUnknown;
    other.m_data = nullptr;
    other.m_refCount = nullptr;
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
    m_refCount = other.m_refCount;

    other.m_size = 0;
    other.m_dims.clear();
    other.m_device = DeviceType::kUnknown;
    other.m_dtype = DType::kUnknown;
    other.m_data = nullptr;
    other.m_refCount = nullptr;
    return *this;
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
    T* hostData = mgr.allocateTyped<T>(DeviceType::kCpu, nullptr, m_size);
    try {
        const cudaError_t err =
            cudaMemcpy(hostData, m_data, sizeof(T) * m_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy(DeviceToHost) failed: ") +
                                     cudaGetErrorString(err));
        }
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
    m_refCount = new std::size_t(1);
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::cuda() {
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
    T* deviceData = mgr.allocateTyped<T>(DeviceType::kCuda, nullptr, m_size);
    try {
        const cudaError_t err =
            cudaMemcpy(deviceData, m_data, sizeof(T) * m_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy(HostToDevice) failed: ") +
                                     cudaGetErrorString(err));
        }
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
    m_refCount = new std::size_t(1);
    return *this;
}

template <typename T>
void Tensor<T>::release() noexcept {
    if (m_data == nullptr) {
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

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;

}  // namespace eUTIL
