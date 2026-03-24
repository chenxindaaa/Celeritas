#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include "udm/memory/MemoryMgr.h"

TEST(test_pool, cpu_allocate_and_release) {
    auto& mgr = eUTIL::MemoryMgr::getInstance();
    constexpr std::size_t kCount = 64;

    int8_t* ptr = mgr.allocateTyped<int8_t>(eUTIL::DeviceType::kCpu, kCount);
    ASSERT_NE(ptr, nullptr);

    mgr.releaseTyped<int8_t>(eUTIL::DeviceType::kCpu, ptr, kCount);
}

TEST(test_pool, cpu_zero_count_returns_nullptr) {
    auto& mgr = eUTIL::MemoryMgr::getInstance();
    EXPECT_EQ(mgr.allocateTyped<int8_t>(eUTIL::DeviceType::kCpu, 0), nullptr);
}

TEST(test_pool, gpu_allocate_and_release) {
    auto& mgr = eUTIL::MemoryMgr::getInstance();
    constexpr std::size_t kCount = 16;

    float* devicePtr = mgr.allocateTyped<float>(eUTIL::DeviceType::kCuda, kCount);
    ASSERT_NE(devicePtr, nullptr);
    ASSERT_EQ(cudaMemset(devicePtr, 0, sizeof(float) * kCount), cudaSuccess);

    mgr.releaseTyped<float>(eUTIL::DeviceType::kCuda, devicePtr, kCount);
}

