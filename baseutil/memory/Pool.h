#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../cudaUtil/utils.hpp"
#include "../designPattren/Singleton.h"

namespace baseutil {
namespace memory {

// Base memory pool with size-class free lists (bucketized by request size).
class MemoryPool {
   public:
    MemoryPool() : m_cachedBytes(0), m_cachedBlockCount(0) {}
    virtual ~MemoryPool() = default;

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = delete;
    MemoryPool& operator=(MemoryPool&&) = delete;

    void* AllocateBytes(std::size_t bytes);
    void DeallocateBytes(void* ptr, std::size_t bytes);

    std::size_t CachedBlockCount() const;
    std::size_t CachedBytes() const;

    template <typename T>
    T* Allocate(std::size_t count = 1) {
        if (count == 0) {
            return nullptr;
        }
        return static_cast<T*>(AllocateBytes(sizeof(T) * count));
    }

    template <typename T>
    void Deallocate(T* ptr, std::size_t count = 1) {
        if (ptr == nullptr) {
            return;
        }
        DeallocateBytes(static_cast<void*>(ptr), sizeof(T) * count);
    }

   protected:
    void ClearCachedBlocks();
    virtual void* AllocateRaw(std::size_t bytes) = 0;
    virtual void DeallocateRaw(void* ptr) = 0;

   private:
    mutable std::mutex m_mutex;
    std::unordered_map<std::size_t, std::vector<void*>> m_freeLists;
    std::size_t m_cachedBytes;
    std::size_t m_cachedBlockCount;
};

class CpuMemoryPool final : public MemoryPool {
   public:
    ~CpuMemoryPool() override { ClearCachedBlocks(); }

   protected:
    void* AllocateRaw(std::size_t bytes) override;
    void DeallocateRaw(void* ptr) override;
};

class CudaMemoryPool final : public MemoryPool {
   public:
    ~CudaMemoryPool() override { ClearCachedBlocks(); }

   protected:
    void* AllocateRaw(std::size_t bytes) override;
    void DeallocateRaw(void* ptr) override;
};

template <typename T>
struct PoolTraits {
    static constexpr bool kSupported = false;
};

template <>
struct PoolTraits<int> {
    static constexpr bool kSupported = true;
};

template <>
struct PoolTraits<float> {
    static constexpr bool kSupported = true;
};

template <>
struct PoolTraits<double> {
    static constexpr bool kSupported = true;
};

template <typename T>
T* AllocateTyped(MemoryPool& pool, std::size_t count = 1) {
    static_assert(PoolTraits<T>::kSupported,
                  "Type is not specialized in PoolTraits");
    return pool.Allocate<T>(count);
}

template <typename T>
void DeallocateTyped(MemoryPool& pool, T* ptr, std::size_t count = 1) {
    static_assert(PoolTraits<T>::kSupported,
                  "Type is not specialized in PoolTraits");
    pool.Deallocate<T>(ptr, count);
}

class MemoryMgr final : public Singleton<MemoryMgr> {
    friend class Singleton<MemoryMgr>;

   public:
    ~MemoryMgr();

    MemoryMgr(const MemoryMgr&) = delete;
    MemoryMgr& operator=(const MemoryMgr&) = delete;

    CpuMemoryPool& GetCpuPool();
    CudaMemoryPool& GetCudaPool();

    // Registry + factory method: new pool types need no MemoryMgr field changes.
    template <typename TPool, typename... Args>
    TPool& GetPool(Args&&... args) {
        static_assert(std::is_base_of<MemoryPool, TPool>::value,
                      "TPool must derive from MemoryPool");

        const std::type_index key(typeid(TPool));
        std::lock_guard<std::mutex> lock(m_poolMutex);
        auto it = m_pools.find(key);
        if (it == m_pools.end()) {
            it = m_pools
                     .emplace(
                         key,
                         std::make_unique<TPool>(std::forward<Args>(args)...))
                     .first;
        }
        return *static_cast<TPool*>(it->second.get());
    }

    void DestroyAllPools();

   private:
    MemoryMgr() = default;

    std::mutex m_poolMutex;
    std::unordered_map<std::type_index, std::unique_ptr<MemoryPool>> m_pools;
};

}  // namespace memory
}  // namespace baseutil
