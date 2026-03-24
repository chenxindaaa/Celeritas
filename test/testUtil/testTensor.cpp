#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <utility>

#include "udm/core/Tensor.h"

using eUTIL::DeviceType;
using eUTIL::DType;
using eUTIL::Tensor;

TEST(test_tensor, cpu_tensor_alloc_and_reuse) {
    constexpr std::size_t kCount = 37;
    int8_t* first_ptr = nullptr;

    {
        Tensor<int8_t> t1(DeviceType::kCpu, kCount);
        EXPECT_EQ(t1.size(), kCount);
        EXPECT_EQ(t1.device(), DeviceType::kCpu);
        ASSERT_NE(t1.data(), nullptr);
        first_ptr = t1.data();
    }

    {
        Tensor<int8_t> t2(DeviceType::kCpu, kCount);
        EXPECT_EQ(t2.size(), kCount);
        EXPECT_EQ(t2.device(), DeviceType::kCpu);
        ASSERT_NE(t2.data(), nullptr);
        EXPECT_EQ(t2.data(), first_ptr);
    }
}

TEST(test_tensor, gpu_tensor_alloc_and_reuse) {
    constexpr std::size_t kCount = 33;
    float* first_ptr = nullptr;

    {
        Tensor<float> t1(DeviceType::kCuda, kCount);
        EXPECT_EQ(t1.size(), kCount);
        EXPECT_EQ(t1.device(), DeviceType::kCuda);
        ASSERT_NE(t1.data(), nullptr);
        ASSERT_EQ(cudaMemset(t1.data(), 0, sizeof(float) * kCount), cudaSuccess);
        first_ptr = t1.data();
    }

    {
        Tensor<float> t2(DeviceType::kCuda, kCount);
        EXPECT_EQ(t2.size(), kCount);
        EXPECT_EQ(t2.device(), DeviceType::kCuda);
        ASSERT_NE(t2.data(), nullptr);
        EXPECT_EQ(t2.data(), first_ptr);
    }
}

TEST(test_tensor, tensor_multi_dims_size) {
    Tensor<int8_t> t(DeviceType::kCpu, 3, 4);
    EXPECT_EQ(t.size(), 12);
    EXPECT_EQ(t.dtype(), DType::kInt8);
    ASSERT_EQ(t.dims().size(), 2);
    EXPECT_EQ(t.dims()[0], 3);
    EXPECT_EQ(t.dims()[1], 4);
}

TEST(test_tensor, tensor_dtype_matches_template_type) {
    Tensor<int8_t> tInt(DeviceType::kCpu, 2, 3);
    Tensor<eUTIL::float16> tHalf(DeviceType::kCpu, 2, 3);
    Tensor<float> tFloat(DeviceType::kCpu, 2, 3);

    EXPECT_EQ(tInt.dtype(), DType::kInt8);
    EXPECT_EQ(tHalf.dtype(), DType::kFloat16);
    EXPECT_EQ(tFloat.dtype(), DType::kFloat32);
}

TEST(test_tensor, tensor_dtype_survives_copy_move_and_device_convert) {
    Tensor<float> a(DeviceType::kCpu, 8);
    EXPECT_EQ(a.dtype(), DType::kFloat32);

    Tensor<float> b = a;
    EXPECT_EQ(a.dtype(), DType::kFloat32);
    EXPECT_EQ(b.dtype(), DType::kFloat32);

    Tensor<float> c = std::move(b);
    EXPECT_EQ(c.dtype(), DType::kFloat32);
    EXPECT_EQ(b.dtype(), DType::kUnknown);

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
        c.cuda();
        EXPECT_EQ(c.device(), DeviceType::kCuda);
        EXPECT_EQ(c.dtype(), DType::kFloat32);

        c.cpu();
        EXPECT_EQ(c.device(), DeviceType::kCpu);
        EXPECT_EQ(c.dtype(), DType::kFloat32);
    }
}

TEST(test_tensor, tensor_copy_increments_refcount) {
    Tensor<int8_t> a(DeviceType::kCpu, 8);
    EXPECT_EQ(a.useCount(), 1);

    Tensor<int8_t> b = a;
    EXPECT_EQ(a.useCount(), 2);
    EXPECT_EQ(b.useCount(), 2);
}

TEST(test_tensor, tensor_copy_assignment_increments_refcount) {
    Tensor<int8_t> a(DeviceType::kCpu, 8);
    Tensor<int8_t> b(DeviceType::kCpu, 8);
    b = a;

    EXPECT_EQ(a.useCount(), 2);
    EXPECT_EQ(b.useCount(), 2);
    EXPECT_EQ(a.data(), b.data());
}

TEST(test_tensor, tensor_move_transfers_ownership) {
    Tensor<int8_t> a(DeviceType::kCpu, 8);
    int8_t* raw = a.data();
    Tensor<int8_t> b = std::move(a);

    EXPECT_EQ(b.useCount(), 1);
    EXPECT_EQ(b.data(), raw);
    EXPECT_EQ(a.useCount(), 0);
}

TEST(test_tensor, tensor_invalid_dims_throw) {
    EXPECT_THROW((Tensor<int8_t>(DeviceType::kCpu, {0, 4})), std::invalid_argument);
    EXPECT_THROW((Tensor<int8_t>(DeviceType::kCpu, {})), std::invalid_argument);
}

TEST(test_tensor, tensor_cuda_breaks_cpu_sharing) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        GTEST_SKIP() << "CUDA device not available";
    }

    Tensor<float> cpuA(DeviceType::kCpu, 8);
    Tensor<float> cpuB = cpuA;
    ASSERT_EQ(cpuA.useCount(), 2);
    ASSERT_EQ(cpuB.useCount(), 2);

    cpuA.cuda();
    EXPECT_EQ(cpuA.device(), DeviceType::kCuda);
    EXPECT_EQ(cpuB.device(), DeviceType::kCpu);
    EXPECT_EQ(cpuA.useCount(), 1);
    EXPECT_EQ(cpuB.useCount(), 1);
}

TEST(test_tensor, tensor_clone_creates_deep_copy_on_cpu) {
    Tensor<int8_t> source(DeviceType::kCpu, 2, 3);
    for (int i = 0; i < 6; ++i) {
        source[i] = static_cast<int8_t>(i + 1);
    }

    Tensor<int8_t> cloned = source.clone();

    EXPECT_NE(cloned.data(), source.data());
    EXPECT_EQ(cloned.device(), DeviceType::kCpu);
    EXPECT_EQ(cloned.dims(), source.dims());
    EXPECT_EQ(cloned.useCount(), 1);
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(cloned[i], source[i]);
    }

    source[0] = 99;
    EXPECT_EQ(static_cast<int>(cloned[0]), 1);
}

TEST(test_tensor, tensor_clone_preserves_cuda_data) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        GTEST_SKIP() << "CUDA device not available";
    }

    Tensor<float> cpu(DeviceType::kCpu, 4);
    for (int i = 0; i < 4; ++i) {
        cpu[i] = static_cast<float>(i) + 0.5f;
    }

    Tensor<float> cpuRef = cpu.clone();
    Tensor<float> gpu = cpu.cuda();
    Tensor<float> gpuClone = gpu.clone();

    EXPECT_NE(gpuClone.data(), gpu.data());
    EXPECT_EQ(gpuClone.device(), DeviceType::kCuda);
    EXPECT_EQ(gpuClone.dims(), gpu.dims());
    EXPECT_EQ(gpuClone.useCount(), 1);

    gpuClone.cpu();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(gpuClone[i], cpuRef[i]);
    }
}

TEST(test_tensor, tensor_external_data_is_non_owning) {
    constexpr std::size_t kCount = 8;
    int8_t* raw = new int8_t[kCount];
    for (std::size_t i = 0; i < kCount; ++i) {
        raw[i] = static_cast<int8_t>(i);
    }

    {
        Tensor<int8_t> a(DeviceType::kCpu, {kCount}, raw, true);
        Tensor<int8_t> b = a;

        EXPECT_EQ(a.data(), raw);
        EXPECT_EQ(b.data(), raw);
        EXPECT_EQ(a.useCount(), 2);
        EXPECT_EQ(b.useCount(), 2);
        for (std::size_t i = 0; i < kCount; ++i) {
            EXPECT_EQ(a[static_cast<int>(i)], raw[i]);
        }
    }

    delete[] raw;
}

TEST(test_tensor, tensor_non_external_data_is_deep_copied) {
    constexpr std::size_t kCount = 8;
    int8_t* raw = new int8_t[kCount];
    for (std::size_t i = 0; i < kCount; ++i) {
        raw[i] = static_cast<int8_t>(i + 1);
    }

    Tensor<int8_t> copied(DeviceType::kCpu, {kCount}, raw, false);

    ASSERT_NE(copied.data(), nullptr);
    EXPECT_NE(copied.data(), raw);
    EXPECT_EQ(copied.useCount(), 1);
    for (std::size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(copied[static_cast<int>(i)], raw[i]);
    }

    raw[0] = 99;
    EXPECT_EQ(static_cast<int>(copied[0]), 1);

    delete[] raw;
}

