#include "MemoryMgr.h"

namespace eUTIL {

MemoryMgr::~MemoryMgr() {
    destroyAllPools();
}

void MemoryMgr::shutdown() {
    destroyAllPools();
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
