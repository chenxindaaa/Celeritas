#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <string>
#include <type_traits>
#include <vector>
#include <assert.h>

#include "../common/BaseTypes.h"
#include "../utils/utils.hpp"

namespace eUTIL {

template <typename T>
class Tensor {
public:
    static_assert(DTypeTrait<T>::kValue != DType::kUnknown,
                    "Unsupported tensor dtype");

    // Creates an empty tensor with unknown device and dtype.
    Tensor() noexcept;
    explicit Tensor(DeviceType device);
    Tensor(DeviceType device, std::initializer_list<std::size_t> dims);
    Tensor(DeviceType device,
           std::initializer_list<std::size_t> dims,
           T* data,
           bool isExternal);

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

    Tensor clone() const;

    T& operator[](int idx) { return m_data[idx]; }

    // Convert in-place to CPU/CUDA storage. If already on target device, no-op.
    Tensor& cpu();
    Tensor& cuda(cudaStream_t stream = nullptr);
    std::string toString() const;

    T* data() { return m_data; }
    const T* data() const { return m_data; }
    int32_t dimSize() const { return static_cast<int32_t>(m_dims.size()); }
    const std::vector<std::size_t>& dims() const { return m_dims; }
    int32_t getDim(int idx) const { assert(idx < m_dims.size()); return m_dims[idx]; }
    bool empty() const { return !m_size || !m_data; }
    DeviceType device() const { return m_device; }
    DType dtype() const { return m_dtype; }
    std::size_t size() const { return m_size; }
    std::size_t byteSize() const { return m_size * sizeof(T); }
    std::size_t useCount() const { return m_control ? m_control->refCount : 0; }
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
    struct ControlBlock {
        std::size_t refCount;
        bool isExternal;
    };

    void acquireFrom(const Tensor& other);
    void releaseOwnership() noexcept;
    void release() noexcept;
    static std::size_t calcSize(std::initializer_list<std::size_t> dims);

    std::size_t m_size;
    std::vector<std::size_t> m_dims;
    DeviceType m_device;
    DType m_dtype;
    T* m_data;
    ControlBlock* m_control;
};

extern template class Tensor<int8_t>;
extern template class Tensor<std::size_t>;
extern template class Tensor<float16>;
extern template class Tensor<float>;

template <typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor);

}  // namespace eUTIL
