#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include "Tokenizer.h"
#include "celeritas/tokenizer/BpeTokenizer.h"
#include "celeritas/tokenizer/HfTokenizerAdapter.h"
#include "celeritas/tokenizer/SentencePieceTokenizer.h"

namespace eCEL {
namespace {

std::unordered_map<std::string, TokenizerLoader::Factory>& factoryRegistry() {
    static std::unordered_map<std::string, TokenizerLoader::Factory> registry = {
        {"bpe",
         [](const std::string& path) {
             return std::make_unique<BpeTokenizer>(path);
         }},
        {"sentencepiece",
         [](const std::string& path) {
             return std::make_unique<SentencePieceTokenizer>(path);
         }},
        {"hf",
         [](const std::string& path) {
             return std::make_unique<HfTokenizerAdapter>(path);
         }},
    };
    return registry;
}

std::string normalizeType(std::string type) {
    std::transform(type.begin(),
                   type.end(),
                   type.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return type;
}

bool endsWith(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
}

}  // namespace

StreamingDetokenizer::StreamingDetokenizer(const Tokenizer& tokenizer,
                                           DecodeOptions options)
    : m_tokenizer(&tokenizer),
      m_options(std::move(options)) {}

std::string StreamingDetokenizer::push(TokenId id) {
    return push(std::vector<TokenId>{id});
}

std::string StreamingDetokenizer::push(const std::vector<TokenId>& ids) {
    m_bufferedIds.insert(m_bufferedIds.end(), ids.begin(), ids.end());
    const std::string decoded = m_tokenizer->decode(m_bufferedIds, m_options);
    const std::string delta = decoded.substr(m_emittedText.size());
    m_emittedText = decoded;
    return delta;
}

std::string StreamingDetokenizer::flush() const {
    return m_emittedText;
}

void StreamingDetokenizer::reset() {
    m_bufferedIds.clear();
    m_emittedText.clear();
}

void TokenizerLoader::registerFactory(std::string type, Factory factory) {
    factoryRegistry()[normalizeType(std::move(type))] = std::move(factory);
}

std::unique_ptr<Tokenizer> TokenizerLoader::load(const std::string& type,
                                                 const std::string& path) {
    const std::string normalizedType = normalizeType(type);
    auto& registry = factoryRegistry();
    const auto it = registry.find(normalizedType);
    if (it == registry.end()) {
        throw std::invalid_argument("Unsupported tokenizer type: " + type);
    }
    return it->second(path);
}

std::unique_ptr<Tokenizer> TokenizerLoader::loadFromFile(
    const std::string& path) {
    const std::filesystem::path fsPath(path);
    const std::string filename = normalizeType(fsPath.filename().string());
    const std::string extension = normalizeType(fsPath.extension().string());

    if (extension == ".model" || endsWith(filename, "spiece.model")) {
        return load("sentencepiece", path);
    }

    if (filename == "tokenizer.json" || extension == ".json") {
        return load("hf", path);
    }

    if (extension == ".bpe" || extension == ".vocab" || extension == ".merges") {
        return load("bpe", path);
    }

    throw std::invalid_argument(
        "Unable to infer tokenizer type from file: " + path);
}

ModelInputs ModelInputBuilder::fromIds(const std::vector<TokenId>& ids,
                                       std::optional<TokenId> padToken) {
    ModelInputs inputs;
    inputs.m_inputIds = ids;
    inputs.m_attentionMask.reserve(ids.size());
    inputs.m_positionIds.reserve(ids.size());

    for (size_t i = 0; i < ids.size(); ++i) {
        const bool isPad = padToken.has_value() && ids[i] == *padToken;
        inputs.m_attentionMask.push_back(isPad ? 0 : 1);
        inputs.m_positionIds.push_back(static_cast<int32_t>(i));
    }

    return inputs;
}

ModelInputs ModelInputBuilder::fromText(const Tokenizer& tokenizer,
                                        std::string_view text,
                                        const EncodeOptions& options) {
    const std::vector<TokenId> ids = tokenizer.encode(text, options);
    return fromIds(ids, tokenizer.specialTokens().pad);
}

ModelInputs ModelInputBuilder::singleToken(TokenId id, int32_t position) {
    ModelInputs inputs;
    inputs.m_inputIds = {id};
    inputs.m_attentionMask = {1};
    inputs.m_positionIds = {position};
    return inputs;
}

}  // namespace eCEL
