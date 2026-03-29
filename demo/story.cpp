#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

#include "celeritas/engine/LLMEngine.h"
#include "celeritas/loader/Loader.h"
#include "celeritas/models/Llama2.h"
#include "celeritas/sampler/argmax_sampler.h"
#include "celeritas/tokenizer/SentencePieceTokenizer.h"
#include "udm/common/BaseTypes.h"
#include "udm/memory/MemoryMgr.h"

int main() {
    const std::string checkpointPath = "/home/cxd/Celeritas/models/stories110M.bin";
    const std::string tokenizerPath = "/home/cxd/Celeritas/models/tokenizer.model";
    eUTIL::DeviceType deviceType = eUTIL::DeviceType::kCuda;
    const std::string prompt = "Once upon a time";

    eCEL::GenerationOptions options;
    options.m_maxNewTokens = 128;

    eCEL::Loader loader;
    auto model = loader.load(eCEL::ModelType::kLlama2, checkpointPath, deviceType);
    auto tokenizer = std::make_unique<eCEL::SentencePieceTokenizer>(tokenizerPath);
    auto sampler = std::make_unique<eCEL::ArgmaxSampler>(deviceType);

    eCEL::LLMEngine engine(std::move(model), std::move(tokenizer), std::move(sampler));
    
    auto start = std::chrono::steady_clock::now();
    const eCEL::GenerationResult result = engine.generate(prompt, options);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    printf("\nsteps/s:%lf\n", static_cast<double>(options.m_maxNewTokens) / duration);

    std::cout << "Checkpoint: " << checkpointPath << std::endl;
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Generated tokens: " << result.m_generatedIds.size() << std::endl;
    std::cout << "Story:" << std::endl;
    std::cout << prompt << result.m_generatedText << std::endl;

    eUTIL::MemoryMgr::getInstance().shutdown();
    return 0;
}
