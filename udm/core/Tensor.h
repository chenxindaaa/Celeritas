#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <vector>

#include "../common/BaseTypes.h"
#include "../utils/utils.hpp"

namespace eUTIL {

template <typename T>
class Tensor {
public:
    static_assert(DTypeTrait<T>::kValue != DType::kUnknown,
                    "Unsupported tensor dtype");

    Tensor(DeviceType device, std::initializer_list<std::size_t> dims);

    template <typename... Dims,
                typename = std::enable_if_t<(sizeof...(Dims) > 0) &&
                                            (std::is_integral_v<Dims> && ...)>>
    Tensor(DeviceType device, Dims... dims)
        : Tensor(device, {static_cast<std::size_t>(dims)...}) {}

    ~Tensor() noexcept;

    Tensor(const Tensor&);
    Tensor& operator=(const Tensor&);

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    T& operator[](int idx) { return m_data[idx]; }

    // Convert in-place to CPU/CUDA storage. If already on target device, no-op.
    Tensor& cpu();
    Tensor& cuda(cudaStream_t stream = nullptr);

    T* data() { return m_data; }
    const T* data() const { return m_data; }
    int32_t dimSize() const { return static_cast<int32_t>(m_dims.size()); }
    const std::vector<std::size_t>& dims() const { return m_dims; }
    bool empty() const { return !m_size || !m_data; }
    DeviceType device() const { return m_device; }
    DType dtype() const { return m_dtype; }
    std::size_t size() const { return m_size; }
    std::size_t byteSize() const { return m_size * sizeof(T); }
    std::size_t useCount() const { return m_refCount ? *m_refCount : 0; }
    const std::vector<size_t> strides() const
    {
        std::vector<size_t> strides;
        if (!m_dims.empty()) {
            for (int32_t i = 0; i < m_dims.size() - 1; ++i) {
            size_t stride = reduceDims(m_dims.begin() + i + 1, m_dims.end(), 1);
            strides.push_back(stride);
            }
            strides.push_back(1);
        }
        return strides;
    }

private:
    void acquireFrom(const Tensor& other);
    void releaseOwnership() noexcept;
    void release() noexcept;
    static std::size_t calcSize(std::initializer_list<std::size_t> dims);

    std::size_t m_size;
    std::vector<std::size_t> m_dims;
    DeviceType m_device;
    DType m_dtype;
    T* m_data;
    std::size_t* m_refCount;
};

extern template class Tensor<int>;
extern template class Tensor<float>;
extern template class Tensor<double>;

}  // namespace eUTIL
