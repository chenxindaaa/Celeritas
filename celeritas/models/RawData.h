#pragma once

#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
namespace eCEL {
class RawModelData {
   public:
    virtual ~RawModelData();
    int32_t fd = -1;
    size_t file_size = 0;
    void* data = nullptr;
    void* weight_data = nullptr;

    virtual const void* weight(size_t offset) const = 0;
};

class RawModelDataFp32 : public RawModelData {
   public:
    const void* weight(size_t offset) const override;
};

class RawModelDataInt8 : public RawModelData {
   public:
    const void* weight(size_t offset) const override;
};

inline RawModelData::~RawModelData() {
    if (data != nullptr && file_size > 0) {
        munmap(data, file_size);
    }
    if (fd != -1) {
        close(fd);
    }
}

inline const void* RawModelDataFp32::weight(size_t offset) const {
    return static_cast<const float*>(weight_data) + offset;
}

inline const void* RawModelDataInt8::weight(size_t offset) const {
    return static_cast<const int8_t*>(weight_data) + offset;
}

}  // namespace eCEL
