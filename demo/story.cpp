#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "celeritas/engine/LLMEgine.h"
#include "celeritas/models/Llama2.h"
#include "celeritas/tokenizer/SentencePieceTokenizer.h"

int main() {
    const std::string checkpointPath = "./models/stories110M.bin";
    const std::string tokenizerPath = "./models/tokenizer.model";
    const std::string prompt = "Once upon a time";

    try {
        auto model = std::make_unique<eCEL::Llama2>(checkpointPath, tokenizerPath);
        auto tokenizer = std::make_unique<eCEL::SentencePieceTokenizer>(tokenizerPath);

        eCEL::LLMEngine engine(std::move(model), std::move(tokenizer));
        const eCEL::GenerationResult result = engine.generate(prompt);

        std::cout << "LLMEngine initialized from checkpoint: " << checkpointPath << std::endl;
        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Prompt token ids:";
        for (const eCEL::TokenId id : result.m_promptIds) {
            std::cout << ' ' << id;
        }
        std::cout << std::endl;
        std::cout << "Embedding output size: " << result.m_modelOutput.size() << std::endl;
        std::cout << "Embedding preview:";
        const std::size_t previewCount = std::min<std::size_t>(result.m_modelOutput.size(), 8);
        for (std::size_t i = 0; i < previewCount; ++i) {
            std::cout << ' ' << result.m_modelOutput[i];
        }
        std::cout << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Failed to initialize LLMEngine: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
