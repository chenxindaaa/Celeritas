#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "../designPattren/Singleton.h"
#include "Pool.h"

namespace eUTIL {

class MemoryMgr final : public Singleton<MemoryMgr> {
friend class Singleton<MemoryMgr>;

public:
    ~MemoryMgr();

    MemoryMgr(const MemoryMgr&) = delete;
    MemoryMgr& operator=(const MemoryMgr&) = delete;

    void shutdown();

    template <typename T>
    T* allocateTyped(DeviceType device, T* ptr = nullptr, std::size_t count = 1) {
        static_assert(DTypeTrait<T>::kValue != DType::kUnknown,
                      "allocateTyped only supports DTypeTrait-supported types");

        if (ptr != nullptr) {
            releaseTyped(device, ptr, count);
            return nullptr;
        }

        if (count == 0) {
            return nullptr;
        }

        MemoryPool& pool = getPoolByDevice(device);
        return static_cast<T*>(pool.allocateBytes(sizeof(T) * count));
    }

    template <typename T>
    void releaseTyped(DeviceType device, T* ptr, std::size_t count = 1) {
        static_assert(DTypeTrait<T>::kValue != DType::kUnknown,
                      "releaseTyped only supports DTypeTrait-supported types");

        if (ptr == nullptr || count == 0) {
            return;
        }

        MemoryPool& pool = getPoolByDevice(device);
        pool.deallocateBytes(static_cast<void*>(ptr), sizeof(T) * count);
    }

private:
    MemoryMgr() = default;

    CpuMemoryPool& getCpuPool();
    CudaMemoryPool& getCudaPool();
    MemoryPool& getPoolByDevice(DeviceType device);

    template <typename TPool>
    TPool& getPool() {
        static_assert(std::is_base_of<MemoryPool, TPool>::value,
                      "TPool must derive from MemoryPool");

        constexpr std::size_t index = static_cast<std::size_t>(TPool::kDeviceType);
        std::unique_ptr<MemoryPool>& pool = m_pools[index];
        if (!pool) {
            pool = std::make_unique<TPool>();
        }
        return *static_cast<TPool*>(pool.get());
    }

    void destroyAllPools();

    std::array<std::unique_ptr<MemoryPool>,
               static_cast<std::size_t>(DeviceType::kNumDeviceTypes)>
        m_pools{};
};

}  // namespace eUTIL
