#pragma once

#include <stdexcept>
#include <string>

namespace eCEL {

enum class OpType {
    kUnknown,
    kAdd,
    kEmb,
    kMatmul,
    kMha,
    kNumOpTypes,
};

inline std::string opTypeName(OpType opType) {
    switch (opType) {
        case OpType::kAdd:
            return "add";
        case OpType::kEmb:
            return "emb";
        case OpType::kMatmul:
            return "matmul";
        case OpType::kMha:
            return "mha";
        case OpType::kUnknown:
        case OpType::kNumOpTypes:
        default:
            return "unknown";
    }
}

}  // namespace eCEL
