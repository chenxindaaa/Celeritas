#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace eCEL {

using TokenId = int32_t;

struct EncodeOptions {
    bool m_addBos = false;
    bool m_addEos = false;
};

struct DecodeOptions {
    bool m_skipSpecialTokens = true;
};

struct SpecialTokens {
    std::optional<TokenId> bos;
    std::optional<TokenId> eos;
    std::optional<TokenId> pad;
    std::optional<TokenId> unk;
    std::unordered_map<std::string, TokenId> extra;
};

struct ChatMessage {
    std::string role;
    std::string content;
};

struct ModelInputs {
    std::vector<TokenId> m_inputIds;
    std::vector<int32_t> m_attentionMask;
    std::vector<int32_t> m_positionIds;
};

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::string name() const = 0;
    virtual size_t vocabSize() const = 0;

    virtual std::vector<TokenId> encode(
        std::string_view text,
        const EncodeOptions& options = {}) const = 0;

    virtual std::string decode(
        const std::vector<TokenId>& ids,
        const DecodeOptions& options = {}) const = 0;

    virtual bool isSpecialToken(TokenId id) const = 0;
    virtual const SpecialTokens& specialTokens() const = 0;
};

class StreamingDetokenizer {
public:
    explicit StreamingDetokenizer(const Tokenizer& tokenizer,
                                  DecodeOptions options = {});

    std::string push(TokenId id);
    std::string push(const std::vector<TokenId>& ids);
    std::string flush() const;
    void reset();

    const std::vector<TokenId>& bufferedIds() const { return m_bufferedIds; }

private:
    const Tokenizer* m_tokenizer = nullptr;
    DecodeOptions m_options;
    std::vector<TokenId> m_bufferedIds;
    std::string m_emittedText;
};

class ChatTemplate {
public:
    virtual ~ChatTemplate() = default;

    virtual std::string name() const = 0;
    virtual std::string format(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = false) const = 0;
};

class TokenizerLoader {
public:
    using Factory =
        std::function<std::unique_ptr<Tokenizer>(const std::string& path)>;

    static void registerFactory(std::string type, Factory factory);
    static std::unique_ptr<Tokenizer> load(const std::string& type,
                                           const std::string& path);
    static std::unique_ptr<Tokenizer> loadFromFile(const std::string& path);
};

class ModelInputBuilder {
public:
    static ModelInputs fromIds(
        const std::vector<TokenId>& ids,
        std::optional<TokenId> padToken = std::nullopt);

    static ModelInputs fromText(const Tokenizer& tokenizer,
                                 std::string_view text,
                                 const EncodeOptions& options = {});

    static ModelInputs singleToken(TokenId id, int32_t position = 0);
};

}  // namespace eCEL
