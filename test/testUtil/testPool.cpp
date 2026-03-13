#include <gtest/gtest.h>
// #include <cuda_runtime_api.h>

#include "baseutil/memory/Pool.h"

using eUTIL::AllocateTyped;
using eUTIL::CpuMemoryPool;
using eUTIL::CudaMemoryPool;
using eUTIL::DeallocateTyped;

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