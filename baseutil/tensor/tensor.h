#pragma once

#include <cstddef>
#include <type_traits>

#include "../memory/Pool.h"

namespace eUTIL {

template <typename T>
class Tensor {
   public:
    static_assert(PoolTraits<T>::kSupported,
                  "Unsupported type");

    Tensor(std::size_t size, DeviceType device);
    ~Tensor() noexcept;

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    T& operator[](int idx) { return m_data[idx]; }

    // Convert in-place to CPU/CUDA storage. If already on target device, no-op.
    Tensor& cpu();
    Tensor& cuda();

    T* data() { return m_data; }
    const T* data() const { return m_data; }
    std::size_t size() const { return m_size; }
    DeviceType device() const { return m_device; }

   private:
    void Release() noexcept;

    std::size_t m_size;
    DeviceType m_device;
    T* m_data;
};

extern template class Tensor<int>;
extern template class Tensor<float>;
extern template class Tensor<double>;

}  // namespace eUTIL
