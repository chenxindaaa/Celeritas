#include "tensor/tensor.h"

#include <stdexcept>
#include <string>
#include <utility>
#include "cudaUtil/utils.hpp"

#include <cuda_runtime_api.h>

namespace eUTIL {

template <typename T>
Tensor<T>::Tensor(std::size_t size, DeviceType device)
    : m_size(size), m_device(device), m_data(nullptr) {
    auto& mgr = MemoryMgr::getInstance();
    switch (m_device) {
        case DeviceType::kCpu:
            m_data = AllocateTyped<T>(mgr.GetCpuPool(), m_size);
            break;
        case DeviceType::kCuda:
            m_data = AllocateTyped<T>(mgr.GetCudaPool(), m_size);
            break;
        default:
            throw std::runtime_error("Unsupported device type");
    }
}

template <typename T>
Tensor<T>::~Tensor() noexcept {
    Release();
}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : m_size(other.m_size), m_device(other.m_device), m_data(other.m_data) {
    other.m_size = 0;
    other.m_data = nullptr;
    other.m_device = DeviceType::kCpu;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    Release();
    m_size = other.m_size;
    m_device = other.m_device;
    m_data = other.m_data;

    other.m_size = 0;
    other.m_data = nullptr;
    other.m_device = DeviceType::kCpu;
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::cpu() {
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

    DeallocateTyped<T>(mgr.GetCudaPool(), m_data, m_size);
    m_data = hostData;
    m_device = DeviceType::kCpu;
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::cuda() {
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

    DeallocateTyped<T>(mgr.GetCpuPool(), m_data, m_size);
    m_data = deviceData;
    m_device = DeviceType::kCuda;
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
        } else {
            DeallocateTyped<T>(mgr.GetCudaPool(), m_data, m_size);
        }
    } catch (...) {
        // Destructors must not throw.
    }

    m_data = nullptr;
    m_size = 0;
    m_device = DeviceType::kCpu;
}

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;

}  // namespace eUTIL
