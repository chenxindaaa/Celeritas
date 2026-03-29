#include "memory/MemoryMgr.h"

#include <algorithm>
#include <cstdlib>
#include <new>
#include <string>

#include <cuda_runtime_api.h>

#include "utils/utils.hpp"

namespace eUTIL {

namespace {

constexpr std::size_t kDefaultCpuCacheLimitBytes = 64 * 1024 * 1024;
constexpr std::size_t kDefaultCudaCacheLimitBytes = 256 * 1024 * 1024;
constexpr std::size_t kKeepHotBlocksPerClass = 1;

// e.g roundUp(34, 16) = 48
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

bool isCudaShutdownError(cudaError_t err) {
    return err == cudaErrorCudartUnloading;
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

void MemoryPool::trimCachedBlocks(std::size_t targetCachedBytes) {
    if (m_cachedBytes <= targetCachedBytes) {
        return;
    }

    std::vector<std::size_t> sizeClasses;
    sizeClasses.reserve(m_freeLists.size());
    for (const auto& entry : m_freeLists) {
        if (!entry.second.empty()) {
            sizeClasses.push_back(entry.first);
        }
    }

    std::sort(sizeClasses.begin(), sizeClasses.end(), std::greater<std::size_t>());

    for (const std::size_t classBytes : sizeClasses) {
        auto it = m_freeLists.find(classBytes);
        if (it == m_freeLists.end()) {
            continue;
        }

        auto& blocks = it->second;
        while (m_cachedBytes > targetCachedBytes && blocks.size() > kKeepHotBlocksPerClass) {
            void* ptr = blocks.back();
            blocks.pop_back();
            deallocateRaw(ptr);
            m_cachedBytes -= classBytes;
            m_cachedBlockCount -= 1;
        }
    }

    if (m_cachedBytes > targetCachedBytes) {
        for (const std::size_t classBytes : sizeClasses) {
            auto it = m_freeLists.find(classBytes);
            if (it == m_freeLists.end()) {
                continue;
            }

            auto& blocks = it->second;
            while (m_cachedBytes > targetCachedBytes && !blocks.empty()) {
                void* ptr = blocks.back();
                blocks.pop_back();
                deallocateRaw(ptr);
                m_cachedBytes -= classBytes;
                m_cachedBlockCount -= 1;
            }
        }
    }

    for (auto it = m_freeLists.begin(); it != m_freeLists.end();) {
        if (it->second.empty()) {
            it = m_freeLists.erase(it);
        } else {
            ++it;
        }
    }
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

    if (m_cachedBytes > cacheLimitBytes()) {
        trimCachedBlocks(cacheTrimTargetBytes());
    }
}

std::size_t MemoryPool::cachedBlockCount() const {
    return m_cachedBlockCount;
}

std::size_t MemoryPool::cachedBytes() const {
    return m_cachedBytes;
}

std::size_t MemoryPool::cacheLimitBytes() const {
    return kDefaultCpuCacheLimitBytes;
}

std::size_t MemoryPool::cacheTrimTargetBytes() const {
    return cacheLimitBytes() / 2;
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

std::size_t CpuMemoryPool::cacheLimitBytes() const {
    return kDefaultCpuCacheLimitBytes;
}

void* CudaMemoryPool::allocateRaw(std::size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void CudaMemoryPool::deallocateRaw(void* ptr) {
    if (ptr != nullptr) {
        const cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess && !isCudaShutdownError(err)) {
            CUDA_CHECK(err);
        }
    }
}

std::size_t CudaMemoryPool::cacheLimitBytes() const {
    return kDefaultCudaCacheLimitBytes;
}

}  // namespace eUTIL
