#include "memory/Pool.h"

#include <cstdlib>
#include <new>
#include <string>
#include <cuda_runtime_api.h>

namespace baseutil {
namespace memory {

namespace {

std::size_t RoundUp(std::size_t bytes, std::size_t alignment) {
    if (alignment == 0) {
        return bytes;
    }
    const std::size_t remainder = bytes % alignment;
    if (remainder == 0) {
        return bytes;
    }
    return bytes + (alignment - remainder);
}

// Bucket requests into size classes to improve freelist hit rate when
// allocation sizes are close but not identical.
std::size_t SizeClassFor(std::size_t bytes) {
    if (bytes <= 256) {
        return RoundUp(bytes, 16);
    }
    if (bytes <= 4 * 1024) {
        return RoundUp(bytes, 128);
    }
    if (bytes <= 64 * 1024) {
        return RoundUp(bytes, 1024);
    }
    return RoundUp(bytes, 4 * 1024);
}

}  // namespace

MemoryMgr::~MemoryMgr() {
    DestroyAllPools();
}

CpuMemoryPool& MemoryMgr::GetCpuPool() {
    return GetPool<CpuMemoryPool>();
}

CudaMemoryPool& MemoryMgr::GetCudaPool() {
    return GetPool<CudaMemoryPool>();
}

void MemoryMgr::DestroyAllPools() {
    std::lock_guard<std::mutex> lock(m_poolMutex);
    m_pools.clear();
}

void MemoryPool::ClearCachedBlocks() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& entry : m_freeLists) {
        for (void* ptr : entry.second) {
            DeallocateRaw(ptr);
        }
    }
    m_freeLists.clear();
    m_cachedBytes = 0;
    m_cachedBlockCount = 0;
}

void* MemoryPool::AllocateBytes(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    const std::size_t classBytes = SizeClassFor(bytes);

    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_freeLists.find(classBytes);
    if (it != m_freeLists.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        m_cachedBytes -= classBytes;
        m_cachedBlockCount -= 1;
        return ptr;
    }

    return AllocateRaw(classBytes);
}

void MemoryPool::DeallocateBytes(void* ptr, std::size_t bytes) {
    if (ptr == nullptr || bytes == 0) {
        return;
    }
    const std::size_t classBytes = SizeClassFor(bytes);

    std::lock_guard<std::mutex> lock(m_mutex);
    m_freeLists[classBytes].push_back(ptr);
    m_cachedBytes += classBytes;
    m_cachedBlockCount += 1;
}

std::size_t MemoryPool::CachedBlockCount() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_cachedBlockCount;
}

std::size_t MemoryPool::CachedBytes() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_cachedBytes;
}

void* CpuMemoryPool::AllocateRaw(std::size_t bytes) {
    void* ptr = std::malloc(bytes);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void CpuMemoryPool::DeallocateRaw(void* ptr) {
    std::free(ptr);
}

void* CudaMemoryPool::AllocateRaw(std::size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void CudaMemoryPool::DeallocateRaw(void* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

template <>
int* AllocateTyped<int>(MemoryPool& pool, std::size_t count) {
    return pool.Allocate<int>(count);
}

template <>
float* AllocateTyped<float>(MemoryPool& pool, std::size_t count) {
    return pool.Allocate<float>(count);
}

template <>
double* AllocateTyped<double>(MemoryPool& pool, std::size_t count) {
    return pool.Allocate<double>(count);
}

template <>
void DeallocateTyped<int>(MemoryPool& pool, int* ptr, std::size_t count) {
    pool.Deallocate<int>(ptr, count);
}

template <>
void DeallocateTyped<float>(MemoryPool& pool, float* ptr,
                            std::size_t count) {
    pool.Deallocate<float>(ptr, count);
}

template <>
void DeallocateTyped<double>(MemoryPool& pool, double* ptr,
                             std::size_t count) {
    pool.Deallocate<double>(ptr, count);
}

}  // namespace memory
}  // namespace baseutil
