#include "MemoryMgr.h"

#include <cstring>

#include "../utils/utils.hpp"

namespace eUTIL {

MemoryMgr::~MemoryMgr() {
    destroyAllPools();
}

void MemoryMgr::shutdown() {
    destroyAllPools();
}

void MemoryMgr::memcpy(DeviceType dstDevice,
                       void* dst,
                       DeviceType srcDevice,
                       const void* src,
                       std::size_t bytes,
                       cudaStream_t stream) {
    if (dst == nullptr || src == nullptr || bytes == 0) {
        return;
    }

    if (dstDevice == DeviceType::kCpu && srcDevice == DeviceType::kCpu) {
        std::memcpy(dst, src, bytes);
        return;
    }

    cudaMemcpyKind kind;
    if (dstDevice == DeviceType::kCpu && srcDevice == DeviceType::kCuda) {
        kind = cudaMemcpyDeviceToHost;
    } else if (dstDevice == DeviceType::kCuda && srcDevice == DeviceType::kCpu) {
        kind = cudaMemcpyHostToDevice;
    } else if (dstDevice == DeviceType::kCuda && srcDevice == DeviceType::kCuda) {
        kind = cudaMemcpyDeviceToDevice;
    } else {
        throw std::invalid_argument("unsupported device pair for memory copy");
    }

    if (stream != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, stream));
        return;
    }
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
}

CpuMemoryPool& MemoryMgr::getCpuPool() {
    return getPool<CpuMemoryPool>();
}

CudaMemoryPool& MemoryMgr::getCudaPool() {
    return getPool<CudaMemoryPool>();
}

MemoryPool& MemoryMgr::getPoolByDevice(DeviceType device) {
    switch (device) {
        case DeviceType::kCpu:
            return getCpuPool();
        case DeviceType::kCuda:
            return getCudaPool();
        case DeviceType::kUnknown:
        case DeviceType::kNumDeviceTypes:
        default:
            throw std::invalid_argument("unsupported device type for memory pool");
    }
}

void MemoryMgr::destroyAllPools() {
    for (auto& pool : m_pools) {
        pool.reset();
    }
}

}  // namespace eUTIL
