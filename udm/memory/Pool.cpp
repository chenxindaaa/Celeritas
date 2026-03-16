#include "memory/MemoryMgr.h"

#include <cstdlib>
#include <new>
#include <string>

#include <cuda_runtime_api.h>

#include "utils/utils.hpp"

namespace eUTIL {

namespace {

std::size_t roundUp(std::size_t bytes, std::size_t alignment) {
    if (alignment == 0) {
        return bytes;
    }
    const std::size_t remainder = bytes % alignment;
    if (remainder == 0) {
        return bytes;
    }
    return bytes + (alignment - remainder);
}

std::size_t sizeClassFor(std::size_t bytes) {
    if (bytes <= 256) {
        return roundUp(bytes, 16);
    }
    if (bytes <= 4 * 1024) {
        return roundUp(bytes, 128);
    }
    if (bytes <= 64 * 1024) {
        return roundUp(bytes, 1024);
    }
    return roundUp(bytes, 4 * 1024);
}

}  // namespace

void MemoryPool::clearCachedBlocks() {
    for (auto& entry : m_freeLists) {
        for (void* ptr : entry.second) {
            deallocateRaw(ptr);
        }
    }
    m_freeLists.clear();
    m_cachedBytes = 0;
    m_cachedBlockCount = 0;
}

void* MemoryPool::allocateBytes(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    const std::size_t classBytes = sizeClassFor(bytes);

    auto it = m_freeLists.find(classBytes);
    if (it != m_freeLists.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        m_cachedBytes -= classBytes;
        m_cachedBlockCount -= 1;
        return ptr;
    }

    return allocateRaw(classBytes);
}

void MemoryPool::deallocateBytes(void* ptr, std::size_t bytes) {
    if (ptr == nullptr || bytes == 0) {
        return;
    }
    const std::size_t classBytes = sizeClassFor(bytes);

    m_freeLists[classBytes].push_back(ptr);
    m_cachedBytes += classBytes;
    m_cachedBlockCount += 1;
}

std::size_t MemoryPool::cachedBlockCount() const {
    return m_cachedBlockCount;
}

std::size_t MemoryPool::cachedBytes() const {
    return m_cachedBytes;
}

void* CpuMemoryPool::allocateRaw(std::size_t bytes) {
    void* ptr = std::malloc(bytes);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void CpuMemoryPool::deallocateRaw(void* ptr) {
    std::free(ptr);
}

void* CudaMemoryPool::allocateRaw(std::size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void CudaMemoryPool::deallocateRaw(void* ptr) {
    if (ptr != nullptr) {
        const cudaError_t err = cudaFree(ptr);
        CUDA_CHECK(err);
    }
}

}  // namespace eUTIL
