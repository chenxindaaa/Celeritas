#pragma once

#include <cstdint>

#include <cuda_fp16.h>

namespace eUTIL {

using float16 = __half;

enum class DeviceType {
    kUnknown,
    kCpu,
    kCuda,
    kNumDeviceTypes,
};

enum class DType {
    kUnknown,
    kInt8,
    kFloat16,
    kFloat32,
    kNumDTypes,
};

template <typename T>
struct DTypeTrait {
    static constexpr DType kValue = DType::kUnknown;
};

template <>
struct DTypeTrait<int8_t> {
    static constexpr DType kValue = DType::kInt8;
};

template <>
struct DTypeTrait<float16> {
    static constexpr DType kValue = DType::kFloat16;
};

template <>
struct DTypeTrait<float> {
    static constexpr DType kValue = DType::kFloat32;
};

inline constexpr bool isUnknownDevice(DeviceType device) {
    return device == DeviceType::kUnknown;
}

inline constexpr bool isCpuDevice(DeviceType device) {
    return device == DeviceType::kCpu;
}

inline constexpr bool isCudaDevice(DeviceType device) {
    return device == DeviceType::kCuda;
}

}  // namespace eUTIL
