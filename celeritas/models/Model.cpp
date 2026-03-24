#include "Model.h"

namespace eCEL{

void Model::init()
{
    readCheckPoint();
    createLayers();
}

void Model::readCheckPoint()
{
    auto fileData = std::make_unique<model::RawModelDataFp32>();
    fileData->fd = open(m_checkPointPath.c_str(), O_RDONLY);
    if (fileData->fd == -1) {
        throw std::runtime_error("failed to open checkpoint: " + m_checkPointPath);
    }

    const off_t fileSize = lseek(fileData->fd, 0, SEEK_END);
    if (fileSize == -1) {
        throw std::runtime_error("failed to get checkpoint size: " + m_checkPointPath);
    }
    if (lseek(fileData->fd, 0, SEEK_SET) == -1) {
        throw std::runtime_error("failed to reset checkpoint offset: " + m_checkPointPath);
    }

    fileData->file_size = static_cast<size_t>(fileSize);
    fileData->data = mmap(nullptr, fileData->file_size, PROT_READ, MAP_PRIVATE, fileData->fd, 0);
    if (fileData->data == MAP_FAILED) {
        fileData->data = nullptr;
        throw std::runtime_error("failed to mmap checkpoint: " + m_checkPointPath);
    }
    if (fileData->file_size < sizeof(ModelConfig)) {
        throw std::runtime_error("checkpoint is smaller than ModelConfig: " + m_checkPointPath);
    }

    std::memcpy(&m_config, fileData->data, sizeof(ModelConfig));
    fileData->weight_data = static_cast<int8_t*>(fileData->data) + sizeof(ModelConfig);
    m_rawData = std::move(fileData);
}

}  // namespace eCEL
