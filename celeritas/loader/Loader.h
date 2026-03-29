#pragma once

// Loads checkpoint data, creates a concrete model instance, and prepares its layers.

#include <memory>
#include <string>

#include "celeritas/models/Model.h"

namespace eCEL {

enum class ModelType {
    kLlama2,
};

class Loader {
public:
    Loader() = default;

    std::unique_ptr<Model> load(ModelType modelType,
                                const std::string& checkpointPath,
                                eUTIL::DeviceType device = eUTIL::DeviceType::kCpu) const;

private:
    // Reads checkpoint bytes and converts them into runtime model state.
    static std::pair<TransformerConfig, std::unique_ptr<RawModelData>> readCheckpoint(
        const std::string& checkpointPath);
};

}  // namespace eCEL
