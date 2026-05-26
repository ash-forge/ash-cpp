// Minimal generation test
#include "inference.h"
#include "logger.h"
#include <iostream>

int main(int argc, char** argv) {
    using namespace ash;
    
    if (argc < 2) {
        std::cerr << "Usage: test_gen <gguf_path> [prompt]\n";
        return 1;
    }
    
    std::string model_path = argv[1];
    
    Logger::instance().set_min_level(LogLevel::INFO);
    
    std::cout << "Loading model...\n";
    InferenceEngine engine;
    if (!engine.load_model(model_path)) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    std::cout << "✅ Model loaded!\n\n";
    
    // Use provided prompt or default
    std::string user_input = (argc >= 3) ? argv[2] : "Write a hello world program";
    std::string prompt =
        "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
        "<|im_start|>user\n" + user_input + "<|im_end|>\n"
        "<|im_start|>assistant\n";
    
    std::cout << "Prompt: \"" << user_input << "\"\n\n";
    std::cout << "Generating...\n" << std::flush;
    
    SamplingConfig config;
    config.max_tokens = 200;
    config.temperature = 0.0f;   // Greedy
    config.use_sampling = false;
    config.repetition_penalty = 1.1f;  // Light penalty to prevent loops
    
    auto result = engine.generate(prompt, config);
    
    std::cout << "\nGenerated text:\n\"" << result.text << "\"\n";
    std::cout << "\nTokens: " << result.tokens_generated << "\n";
    std::cout << "Stop reason: " << result.stop_reason << "\n";
    
    return 0;
}

