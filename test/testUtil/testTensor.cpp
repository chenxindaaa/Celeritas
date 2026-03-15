#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <utility>

#include "baseutil/tensor/tensor.h"

using eUTIL::DeviceType;
using eUTIL::DType;
using eUTIL::Tensor;

TEST(test_tensor, cpu_tensor_alloc_and_reuse) {
    constexpr std::size_t kCount = 37;
    int* first_ptr = nullptr;

    {
        Tensor<int> t1(DeviceType::kCpu, kCount);
        EXPECT_EQ(t1.size(), kCount);
        EXPECT_EQ(t1.device(), DeviceType::kCpu);
        ASSERT_NE(t1.data(), nullptr);
        first_ptr = t1.data();
    }

    {
        Tensor<int> t2(DeviceType::kCpu, kCount);
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
    Tensor<int> t(DeviceType::kCpu, 3, 4);
    EXPECT_EQ(t.size(), 12);
    EXPECT_EQ(t.dtype(), DType::kInt32);
    ASSERT_EQ(t.dims().size(), 2);
    EXPECT_EQ(t.dims()[0], 3);
    EXPECT_EQ(t.dims()[1], 4);
}

TEST(test_tensor, tensor_dtype_matches_template_type) {
    Tensor<int> tInt(DeviceType::kCpu, 2, 3);
    Tensor<float> tFloat(DeviceType::kCpu, 2, 3);
    Tensor<double> tDouble(DeviceType::kCpu, 2, 3);

    EXPECT_EQ(tInt.dtype(), DType::kInt32);
    EXPECT_EQ(tFloat.dtype(), DType::kFloat32);
    EXPECT_EQ(tDouble.dtype(), DType::kFloat64);
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
    Tensor<int> a(DeviceType::kCpu, 8);
    EXPECT_EQ(a.useCount(), 1);

    Tensor<int> b = a;
    EXPECT_EQ(a.useCount(), 2);
    EXPECT_EQ(b.useCount(), 2);
}

TEST(test_tensor, tensor_copy_assignment_increments_refcount) {
    Tensor<int> a(DeviceType::kCpu, 8);
    Tensor<int> b(DeviceType::kCpu, 8);
    b = a;

    EXPECT_EQ(a.useCount(), 2);
    EXPECT_EQ(b.useCount(), 2);
    EXPECT_EQ(a.data(), b.data());
}

TEST(test_tensor, tensor_move_transfers_ownership) {
    Tensor<int> a(DeviceType::kCpu, 8);
    int* raw = a.data();
    Tensor<int> b = std::move(a);

    EXPECT_EQ(b.useCount(), 1);
    EXPECT_EQ(b.data(), raw);
    EXPECT_EQ(a.useCount(), 0);
}

TEST(test_tensor, tensor_invalid_dims_throw) {
    EXPECT_THROW((Tensor<int>(DeviceType::kCpu, {0, 4})), std::invalid_argument);
    EXPECT_THROW((Tensor<int>(DeviceType::kCpu, {})), std::invalid_argument);
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
