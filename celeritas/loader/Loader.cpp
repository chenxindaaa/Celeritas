#include "Loader.h"

#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <utility>

#include "celeritas/models/Llama2.h"

namespace eCEL {
namespace {

constexpr uint32_t kCheckpointMagic = 0x616b3432;
constexpr std::size_t kVersionedHeaderSize = 256;

TransformerConfig makeTransformerConfig(const ModelConfig& rawConfig) {
    TransformerConfig config;
    config.dim = rawConfig.dim;
    config.hiddenDim = rawConfig.hiddenDim;
    config.layerNum = rawConfig.layerNum;
    config.headNum = rawConfig.headNum;
    config.kvHeadNum = rawConfig.kvHeadNum;
    config.seqLen = rawConfig.seqLen;
    config.kvDim = (rawConfig.dim * rawConfig.kvHeadNum) / rawConfig.headNum;
    config.kvMul = rawConfig.headNum / rawConfig.kvHeadNum;
    config.headSize = rawConfig.dim / rawConfig.headNum;
#if defined(QWEN3_SUPPORT)
    config.immediateDim = rawConfig.immediateDim;
#endif
    config.vocabSize = std::abs(rawConfig.vocabSize);
    return config;
}

}  // namespace

std::unique_ptr<Model> Loader::load(ModelType modelType,
                                    const std::string& checkpointPath,
                                    eUTIL::DeviceType device) const {
    std::unique_ptr<Model> model;
    switch (modelType) {
        case ModelType::kLlama2:
            model = std::make_unique<Llama2>(checkpointPath, device);
            break;
        default:
            throw std::invalid_argument("Unsupported model type in Loader::load");
    }

    auto [config, rawData] = readCheckpoint(checkpointPath);
    model->setLoadedState(std::move(config), std::move(rawData));
    model->createLayers();
    model->loadWeights();
    model->to(device);
    return model;
}

std::pair<TransformerConfig, std::unique_ptr<RawModelData>> Loader::readCheckpoint(
    const std::string& checkpointPath) {
    // Open the checkpoint first and keep the file descriptor for mmap-backed lifetime management.
    auto fileData = std::make_unique<RawModelDataFp32>();
    fileData->fd = open(checkpointPath.c_str(), O_RDONLY);
    if (fileData->fd == -1) {
        throw std::runtime_error("failed to open checkpoint: " + checkpointPath);
    }

    // Query the file size up front so we can validate header boundaries before parsing.
    const off_t fileSize = lseek(fileData->fd, 0, SEEK_END);
    if (fileSize == -1) {
        throw std::runtime_error("failed to get checkpoint size: " + checkpointPath);
    }
    if (lseek(fileData->fd, 0, SEEK_SET) == -1) {
        throw std::runtime_error("failed to reset checkpoint offset: " + checkpointPath);
    }

    fileData->file_size = static_cast<size_t>(fileSize);
    // Map the whole checkpoint once and keep raw pointers into the mapped memory.
    fileData->data = mmap(nullptr, fileData->file_size, PROT_READ, MAP_PRIVATE, fileData->fd, 0);
    if (fileData->data == MAP_FAILED) {
        fileData->data = nullptr;
        throw std::runtime_error("failed to mmap checkpoint: " + checkpointPath);
    }
    if (fileData->file_size < sizeof(ModelConfig)) {
        throw std::runtime_error("checkpoint is smaller than supported header: " + checkpointPath);
    }

    uint32_t magic = 0;
    std::memcpy(&magic, fileData->data, sizeof(magic));

    TransformerConfig config;
    if (magic == kCheckpointMagic) {
        // Versioned checkpoints use a fixed-size header with explicit version and flags.
        if (fileData->file_size < kVersionedHeaderSize) {
            throw std::runtime_error("versioned checkpoint header is incomplete: " + checkpointPath);
        }

        int32_t version = 0;
        std::memcpy(&version,
                    static_cast<const int8_t*>(fileData->data) + sizeof(magic),
                    sizeof(version));
        if (version != 1) {
            throw std::runtime_error("unsupported checkpoint version: " + std::to_string(version));
        }

        ModelConfig rawConfig{};
        std::memcpy(&rawConfig,
                    static_cast<const int8_t*>(fileData->data) + sizeof(magic) + sizeof(version),
                    sizeof(ModelConfig));

        config = makeTransformerConfig(rawConfig);
        config.checkpointFormat = CheckpointFormat::kVersion1;

        // Version1 stores whether lm_head shares the embedding table in a dedicated flag byte.
        uint8_t sharedClassifier = 0;
        const std::size_t sharedFlagOffset = sizeof(magic) + sizeof(version) + sizeof(ModelConfig);
        std::memcpy(&sharedClassifier,
                    static_cast<const int8_t*>(fileData->data) + sharedFlagOffset,
                    sizeof(sharedClassifier));
        config.isSharedWeight = sharedClassifier != 0;
        // All version1 weights begin after the padded 256-byte header block.
        fileData->weight_data = static_cast<int8_t*>(fileData->data) + kVersionedHeaderSize;
    } else {
        // Legacy checkpoints begin directly with ModelConfig and encode weight sharing in vocabSize sign.
        ModelConfig rawConfig{};
        std::memcpy(&rawConfig, fileData->data, sizeof(ModelConfig));

        config = makeTransformerConfig(rawConfig);
        config.checkpointFormat = CheckpointFormat::kLegacy;
        config.isSharedWeight = rawConfig.vocabSize > 0;
        // Legacy weights start immediately after the raw config header.
        fileData->weight_data = static_cast<int8_t*>(fileData->data) + sizeof(ModelConfig);
    }

    return {config, std::move(fileData)};
}

}  // namespace eCEL
