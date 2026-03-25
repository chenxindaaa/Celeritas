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

    ModelConfig config{};
    std::memcpy(&config, fileData->data, sizeof(ModelConfig));

    m_config.dim_ = config.dim;
    m_config.hidden_dim_ = config.hidden_dim;
    m_config.layer_num_ = config.layer_num;
    m_config.head_num_ = config.head_num;
    m_config.kv_head_num_ = config.kv_head_num;
    m_config.seq_len_ = config.seq_len;

    m_config.kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
    m_config.kv_mul_ = config.head_num / config.kv_head_num;
    m_config.head_size_ = config.dim / config.head_num;
    #if defined(QWEN3_SUPPORT)
    m_config.immediate_dim_ = config.immediate_dim_;
    #endif
    if (config.vocab_size > 0) {
        m_config.is_shared_weight_ = true;
    } else {
        m_config.is_shared_weight_ = false;
    }
    m_config.vocab_size_ = std::abs(config.vocab_size);
    
    fileData->weight_data = static_cast<int8_t*>(fileData->data) + sizeof(ModelConfig);
    m_rawData = std::move(fileData);
}

}  // namespace eCEL
