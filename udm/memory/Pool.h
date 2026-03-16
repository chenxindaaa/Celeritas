#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../common/BaseTypes.h"

namespace eUTIL {

class MemoryMgr;

class MemoryPool {
friend class ::eUTIL::MemoryMgr;

public:
    explicit MemoryPool(DeviceType device)
        : m_device(device), m_cachedBytes(0), m_cachedBlockCount(0) {}
    virtual ~MemoryPool() = default;

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = delete;
    MemoryPool& operator=(MemoryPool&&) = delete;

    DeviceType device() const { return m_device; }

protected:
    void* allocateBytes(std::size_t bytes);
    void deallocateBytes(void* ptr, std::size_t bytes);
    void clearCachedBlocks();

    std::size_t cachedBlockCount() const;
    std::size_t cachedBytes() const;

    virtual void* allocateRaw(std::size_t bytes) = 0;
    virtual void deallocateRaw(void* ptr) = 0;

    DeviceType m_device;
    std::unordered_map<std::size_t, std::vector<void*>> m_freeLists;
    std::size_t m_cachedBytes;
    std::size_t m_cachedBlockCount;
};

class CpuMemoryPool final : public MemoryPool {
public:
    static constexpr DeviceType kDeviceType = DeviceType::kCpu;

    CpuMemoryPool() : MemoryPool(kDeviceType) {}
    ~CpuMemoryPool() override { clearCachedBlocks(); }

private:
    void* allocateRaw(std::size_t bytes) override;
    void deallocateRaw(void* ptr) override;
};

class CudaMemoryPool final : public MemoryPool {
public:
    static constexpr DeviceType kDeviceType = DeviceType::kCuda;

    CudaMemoryPool() : MemoryPool(kDeviceType) {}
    ~CudaMemoryPool() override { clearCachedBlocks(); }

private:
    void* allocateRaw(std::size_t bytes) override;
    void deallocateRaw(void* ptr) override;
};

}  // namespace eUTIL
