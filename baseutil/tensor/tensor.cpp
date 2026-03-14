#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <cuda_runtime_api.h>

#include "utils/utils.hpp"
#include "tensor/tensor.h"

namespace eUTIL {

template <typename T>
std::size_t Tensor<T>::CalcSize(std::initializer_list<std::size_t> dims) {
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
    : m_size(CalcSize(dims)),
      m_dims(dims),
      m_device(device),
      m_data(nullptr),
      m_refCount(nullptr) {
    if (m_device == DeviceType::kUnknown) {
        throw std::invalid_argument("Tensor device cannot be kUnknown");
    }

    auto& mgr = MemoryMgr::getInstance();
    switch (m_device) {
        case DeviceType::kCpu:
            m_data = AllocateTyped<T>(mgr.GetCpuPool(), m_size);
            break;
        case DeviceType::kCuda:
            m_data = AllocateTyped<T>(mgr.GetCudaPool(), m_size);
            break;
        case DeviceType::kUnknown:
        default:
            throw std::runtime_error("Unsupported device type");
    }

    m_refCount = new std::size_t(1);
}

template <typename T>
Tensor<T>::~Tensor() noexcept {
    ReleaseOwnership();
}

template <typename T>
void Tensor<T>::AcquireFrom(const Tensor& other) {
    m_size = other.m_size;
    m_dims = other.m_dims;
    m_device = other.m_device;
    m_data = other.m_data;
    m_refCount = other.m_refCount;
    if (m_refCount) {
        ++(*m_refCount);
    }
}

template <typename T>
void Tensor<T>::ReleaseOwnership() noexcept {
    if (!m_refCount) {
        return;
    }

    if (--(*m_refCount) == 0) {
        Release();
        delete m_refCount;
    }

    m_refCount = nullptr;
    m_data = nullptr;
    m_size = 0;
    m_dims.clear();
    m_device = DeviceType::kUnknown;
}

template <typename T>
Tensor<T>::Tensor(const Tensor& other)
    : m_size(0),
      m_dims(),
      m_device(DeviceType::kUnknown),
      m_data(nullptr),
      m_refCount(nullptr) {
    AcquireFrom(other);
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }

    // Acquire first for strong exception safety.
    Tensor temp(other);
    ReleaseOwnership();
    AcquireFrom(temp);
    return *this;
}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : m_size(other.m_size),
      m_dims(std::move(other.m_dims)),
      m_device(other.m_device),
      m_data(other.m_data),
      m_refCount(other.m_refCount) {
    other.m_size = 0;
    other.m_dims.clear();
    other.m_device = DeviceType::kUnknown;
    other.m_data = nullptr;
    other.m_refCount = nullptr;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    ReleaseOwnership();
    m_size = other.m_size;
    m_dims = std::move(other.m_dims);
    m_device = other.m_device;
    m_data = other.m_data;
    m_refCount = other.m_refCount;

    other.m_size = 0;
    other.m_dims.clear();
    other.m_device = DeviceType::kUnknown;
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
    T* hostData = AllocateTyped<T>(mgr.GetCpuPool(), m_size);
    try {
        const cudaError_t err =
            cudaMemcpy(hostData, m_data, sizeof(T) * m_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy(DeviceToHost) failed: ") +
                                     cudaGetErrorString(err));
        }
    } catch (...) {
        DeallocateTyped<T>(mgr.GetCpuPool(), hostData, m_size);
        throw;
    }

    const std::size_t originalSize = m_size;
    const std::vector<std::size_t> originalDims = m_dims;
    ReleaseOwnership();
    m_data = hostData;
    m_size = originalSize;
    m_dims = originalDims;
    m_device = DeviceType::kCpu;
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
    T* deviceData = AllocateTyped<T>(mgr.GetCudaPool(), m_size);
    try {
        const cudaError_t err =
            cudaMemcpy(deviceData, m_data, sizeof(T) * m_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy(HostToDevice) failed: ") +
                                     cudaGetErrorString(err));
        }
    } catch (...) {
        DeallocateTyped<T>(mgr.GetCudaPool(), deviceData, m_size);
        throw;
    }

    const std::size_t originalSize = m_size;
    const std::vector<std::size_t> originalDims = m_dims;
    ReleaseOwnership();
    m_data = deviceData;
    m_size = originalSize;
    m_dims = originalDims;
    m_device = DeviceType::kCuda;
    m_refCount = new std::size_t(1);
    return *this;
}

template <typename T>
void Tensor<T>::Release() noexcept {
    if (m_data == nullptr) {
        return;
    }

    try {
        auto& mgr = MemoryMgr::getInstance();
        if (m_device == DeviceType::kCpu) {
            DeallocateTyped<T>(mgr.GetCpuPool(), m_data, m_size);
        } else if (m_device == DeviceType::kCuda) {
            DeallocateTyped<T>(mgr.GetCudaPool(), m_data, m_size);
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
