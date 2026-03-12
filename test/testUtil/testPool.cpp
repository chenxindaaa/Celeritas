#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include "baseutil/memory/Pool.h"
#include "baseutil/tensor/tensor.h"

using baseutil::memory::AllocateTyped;
using baseutil::memory::CpuMemoryPool;
using baseutil::memory::CudaMemoryPool;
using baseutil::memory::DeallocateTyped;
using baseutil::tensor::DeviceType;
using baseutil::tensor::Tensor;

TEST(test_pool, cpu_pool_cache_and_reuse) {
    CpuMemoryPool pool;

    constexpr std::size_t kCount = 64;
    constexpr std::size_t kBytes = sizeof(int) * kCount;

    EXPECT_EQ(pool.CachedBlockCount(), 0u);
    EXPECT_EQ(pool.CachedBytes(), 0u);

    int* first = AllocateTyped<int>(pool, kCount);
    ASSERT_NE(first, nullptr);

    DeallocateTyped<int>(pool, first, kCount);
    EXPECT_EQ(pool.CachedBlockCount(), 1u);
    EXPECT_EQ(pool.CachedBytes(), kBytes);

    int* second = AllocateTyped<int>(pool, kCount);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second, first);
    EXPECT_EQ(pool.CachedBlockCount(), 0u);
    EXPECT_EQ(pool.CachedBytes(), 0u);

    DeallocateTyped<int>(pool, second, kCount);
}

TEST(test_pool, cpu_pool_zero_count_returns_nullptr) {
    CpuMemoryPool pool;
    EXPECT_EQ(pool.Allocate<int>(0), nullptr);
}

TEST(test_pool, gpu_pool_basic) {
    CudaMemoryPool pool;

    constexpr std::size_t kCount = 16;
    float* device_ptr = AllocateTyped<float>(pool, kCount);
    ASSERT_NE(device_ptr, nullptr);

    ASSERT_EQ(cudaMemset(device_ptr, 0, sizeof(float) * kCount), cudaSuccess);
    DeallocateTyped<float>(pool, device_ptr, kCount);

    EXPECT_EQ(pool.CachedBlockCount(), 1u);
    EXPECT_EQ(pool.CachedBytes(), sizeof(float) * kCount);
}

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
