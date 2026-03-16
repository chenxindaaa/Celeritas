#pragma once

namespace eUTIL {

enum class DeviceType {
    kUnknown,
    kCpu,
    kCuda,
    kNumDeviceTypes,
};

enum class DType {
    kUnknown,
    kInt32,
    kFloat32,
    kFloat64,
    kNumDTypes,
};

template <typename T>
struct DTypeTrait {
    static constexpr DType kValue = DType::kUnknown;
};

template <>
struct DTypeTrait<int> {
    static constexpr DType kValue = DType::kInt32;
};

template <>
struct DTypeTrait<float> {
    static constexpr DType kValue = DType::kFloat32;
};

template <>
struct DTypeTrait<double> {
    static constexpr DType kValue = DType::kFloat64;
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
