#include <gtest/gtest.h>
// #include <cuda_runtime_api.h>

#include "baseutil/tensor/tensor.h"

using eUTIL::DeviceType;
using eUTIL::Tensor;

TEST(test_tensor, cpu_tensor_alloc_and_reuse) {
    constexpr std::size_t kCount = 37;
    int* first_ptr = nullptr;

    {
        Tensor<int> t1(kCount, DeviceType::kCpu);
        EXPECT_EQ(t1.size(), kCount);
        EXPECT_EQ(t1.device(), DeviceType::kCpu);
        ASSERT_NE(t1.data(), nullptr);
        first_ptr = t1.data();
    }

    {
        Tensor<int> t2(kCount, DeviceType::kCpu);
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
        Tensor<float> t1(kCount, DeviceType::kCuda);
        EXPECT_EQ(t1.size(), kCount);
        EXPECT_EQ(t1.device(), DeviceType::kCuda);
        ASSERT_NE(t1.data(), nullptr);
        ASSERT_EQ(cudaMemset(t1.data(), 0, sizeof(float) * kCount), cudaSuccess);
        first_ptr = t1.data();
    }

    {
        Tensor<float> t2(kCount, DeviceType::kCuda);
        EXPECT_EQ(t2.size(), kCount);
        EXPECT_EQ(t2.device(), DeviceType::kCuda);
        ASSERT_NE(t2.data(), nullptr);
        EXPECT_EQ(t2.data(), first_ptr);
    }
}
