#include "kernelMgr.h"

#include <cstddef>
#include <stdexcept>

#include "gpu/add.cuh"

namespace eCEL {

template <>
kernelMgr::AddKernel
kernelMgr::get_add_kernel<eUTIL::DeviceType::kCuda>() const {
    return add_kernel_cu;
}

template <>
kernelMgr::AddKernel
kernelMgr::get_add_kernel<eUTIL::DeviceType::kCpu>() const {
    return [](const TensorInt& input1, const TensorInt& input2, TensorInt& output,
              void* stream) {
        (void)stream;

        if (!eUTIL::IsCpuDevice(input1.device()) ||
            !eUTIL::IsCpuDevice(input2.device()) ||
            !eUTIL::IsCpuDevice(output.device())) {
            throw std::invalid_argument("CPU add kernel requires CPU tensors");
        }

        if (input1.size() != input2.size() || input1.size() != output.size()) {
            throw std::invalid_argument(
                "input/output size mismatch in add_kernel_cpu");
        }

        const int* in1 = input1.data();
        const int* in2 = input2.data();
        int* out = output.data();
        for (std::size_t i = 0; i < input1.size(); ++i) {
            out[i] = in1[i] + in2[i];
        }
    };
}

}  // namespace eCEL
