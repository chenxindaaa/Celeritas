#pragma once

#include <functional>
#include <type_traits>

#include "baseutil/designPattren/Singleton.h"
#include "baseutil/tensor/tensor.h"

namespace eCEL {

inline constexpr eUTIL::DeviceType CPU = eUTIL::DeviceType::kCpu;
inline constexpr eUTIL::DeviceType CUDA = eUTIL::DeviceType::kCuda;

template <eUTIL::DeviceType Device>
struct KernelBackendFalse : std::false_type {};

class kernelMgr : public Singleton<kernelMgr> {
   public:
    using TensorInt = eUTIL::Tensor<int>;
    using AddKernel = std::function<void(const TensorInt& input1,
                                         const TensorInt& input2,
                                         TensorInt& output, void* stream)>;

    template <eUTIL::DeviceType Device>
    AddKernel get_add_kernel() const {
        static_assert(KernelBackendFalse<Device>::value,
                      "Unsupported backend for get_add_kernel");
        return {};
    }

    template <eUTIL::DeviceType Device>
    AddKernel get_kernel() const {
        return get_add_kernel<Device>();
    }

   private:
    friend class Singleton<kernelMgr>;

    kernelMgr() = default;
    ~kernelMgr() = default;

    kernelMgr(const kernelMgr&) = delete;
    kernelMgr& operator=(const kernelMgr&) = delete;
};

template <>
kernelMgr::AddKernel
kernelMgr::get_add_kernel<eUTIL::DeviceType::kCuda>() const;

template <>
kernelMgr::AddKernel
kernelMgr::get_add_kernel<eUTIL::DeviceType::kCpu>() const;

}  // namespace eCEL
