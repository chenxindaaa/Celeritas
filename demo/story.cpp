#include <exception>
#include <iostream>
#include <string>

#include "celeritas/models/Llama2.h"

int main() {
    const std::string checkpointPath = "./models/stories110M.bin";
    const std::string tokenizerPath = "./models/tokenizer.model";

    try {
        eCEL::Llama2 llama2(std::move(checkpointPath), std::move(tokenizerPath));
        llama2.init();
        std::cout << "Model initialized from checkpoint: " << checkpointPath << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Failed to initialize model: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
