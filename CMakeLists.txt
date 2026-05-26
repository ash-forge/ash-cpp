cmake_minimum_required(VERSION 3.20)
project(ash_engine VERSION 0.1.0 LANGUAGES C CXX)

# C++20 required for coroutines
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# SQLite library
add_library(sqlite3 STATIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/sqlite-amalgamation-3530000/sqlite3.c)
target_include_directories(sqlite3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/sqlite-amalgamation-3530000)

# Source files
set(SOURCES
    src/main.cpp
    src/event_loop.cpp
    src/logger.cpp
    src/model_loader.cpp
    src/emotional_state.cpp
    src/memory_store.cpp
    src/decision_engine.cpp
    src/context_manager.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/inference.cpp
    src/memory_lock.cpp
)

# Executable
add_executable(ash_engine ${SOURCES})

# Model test executable
add_executable(model_test 
    src/model_test.cpp
    src/logger.cpp
    src/model_loader.cpp
    src/gguf_parser.cpp
    src/tensor.cpp
    src/memory_lock.cpp
)

# Emotional state test executable
add_executable(emotional_test
    src/emotional_test.cpp
    src/logger.cpp
    src/emotional_state.cpp
    src/event_loop.cpp
)

# Memory test executable
add_executable(memory_test
    src/memory_test.cpp
    src/logger.cpp
    src/memory_store.cpp
    src/emotional_state.cpp
)

# Include directories
target_include_directories(ash_engine PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(model_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(emotional_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(memory_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Threading support
find_package(Threads REQUIRED)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found — parallelizing matmul/matvec")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(OMP_LIBS OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP NOT found — single-threaded matmul")
    set(OMP_LIBS "")
endif()
target_link_libraries(ash_engine PRIVATE Threads::Threads sqlite3 ${OMP_LIBS})
target_link_libraries(model_test PRIVATE Threads::Threads)
target_link_libraries(emotional_test PRIVATE Threads::Threads)
# Decision test executable
add_executable(decision_test
    src/decision_test.cpp
    src/logger.cpp
    src/memory_store.cpp
    src/emotional_state.cpp
    src/decision_engine.cpp
)

target_include_directories(decision_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# Context test executable
add_executable(context_test
    src/context_test.cpp
    src/logger.cpp
    src/memory_store.cpp
    src/emotional_state.cpp
    src/context_manager.cpp
)

target_include_directories(context_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# Tensor test executable
add_executable(tensor_test
    src/tensor_test.cpp
    src/logger.cpp
    src/tensor.cpp
)

target_include_directories(tensor_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(tensor_test PRIVATE Threads::Threads)

# Matrix operations test executable
add_executable(matrix_ops_test
    src/matrix_ops_test.cpp
    src/logger.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
)

target_include_directories(matrix_ops_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(matrix_ops_test PRIVATE Threads::Threads ${OMP_LIBS})

# GGUF parser test executable
add_executable(gguf_parser_test
    src/gguf_parser_test.cpp
    src/logger.cpp
    src/tensor.cpp
    src/gguf_parser.cpp
)

target_include_directories(gguf_parser_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(gguf_parser_test PRIVATE Threads::Threads)

# Tokenizer test executable
add_executable(tokenizer_test
    src/tokenizer_test.cpp
    src/logger.cpp
    src/tensor.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
)

target_include_directories(tokenizer_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(tokenizer_test PRIVATE Threads::Threads)

# Attention test executable
add_executable(attention_test
    src/attention_test.cpp
    src/logger.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/attention.cpp
)

target_include_directories(attention_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(attention_test PRIVATE Threads::Threads)

# Inference test executable
add_executable(inference_test
    src/inference_test.cpp
    src/logger.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/safetensors_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/inference.cpp
)

target_include_directories(inference_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(inference_test PRIVATE Threads::Threads ${OMP_LIBS})

# Persona test executable
add_executable(persona_test
    src/persona_test.cpp
    src/logger.cpp
    src/emotional_state.cpp
    src/event_loop.cpp
    src/memory_store.cpp
    src/context_manager.cpp
    src/persona.cpp
)

target_include_directories(persona_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(persona_test PRIVATE Threads::Threads sqlite3)

# Memory lock test
add_executable(memory_lock_test
    src/memory_lock_test.cpp
    src/memory_lock.cpp
    src/tensor.cpp
    src/logger.cpp
)

target_include_directories(memory_lock_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(memory_lock_test PRIVATE Threads::Threads)

# Ash CLI - Proof of Life Test Harness
add_executable(ash_cli
    src/ash_cli.cpp
    src/inference.cpp
    src/persona.cpp
    src/emotional_state.cpp
    src/memory_store.cpp
    src/context_manager.cpp
    src/decision_engine.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
    src/event_loop.cpp
)

target_include_directories(ash_cli PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(ash_cli PRIVATE Threads::Threads sqlite3 ${OMP_LIBS})

# Multi-Model Test - Ash-Forge routing system
add_executable(multimodel_test
    src/multimodel_test.cpp
    src/model_loader.cpp
    src/scout_agent.cpp
    src/gguf_parser.cpp
    src/tensor.cpp
    src/memory_lock.cpp
    src/logger.cpp
)

target_include_directories(multimodel_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(multimodel_test PRIVATE Threads::Threads)

# Installation
install(TARGETS ash_engine DESTINATION bin)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

# Test generation executable (minimal, no persona/memory)
add_executable(test_gen
    ../test_gen.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(test_gen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_gen sqlite3 ${OMP_LIBS})

# Test vocab/tokenizer
add_executable(test_vocab
    ../test_vocab.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(test_vocab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_vocab sqlite3)

# Test token byte representation
add_executable(test_token_bytes
    ../test_token_bytes.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(test_token_bytes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_token_bytes sqlite3)

# Test encoding
add_executable(test_encode
    ../test_encode.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(test_encode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_encode sqlite3)

# Dump vocab
add_executable(dump_vocab
    ../dump_vocab.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(dump_vocab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dump_vocab sqlite3)

# Dump GGUF metadata
add_executable(dump_metadata
    ../dump_metadata.cpp
    src/gguf_parser.cpp
    src/tensor.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(dump_metadata PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Check vocab
add_executable(check_vocab
    ../check_vocab.cpp
    src/tokenizer.cpp
    src/gguf_parser.cpp
    src/tensor.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(check_vocab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(check_vocab sqlite3)

# Test with correct Ollama tokens
add_executable(test_ollama_tokens
    ../test_ollama_tokens.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(test_ollama_tokens PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_ollama_tokens sqlite3)

# Debug first token generation
add_executable(debug_first_token
    ../debug_first_token.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(debug_first_token PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(debug_first_token sqlite3)

# Check weights loading
add_executable(check_weights
    ../check_weights.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(check_weights PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(check_weights sqlite3)

# Check token ranking
add_executable(check_ranking
    check_ranking.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(check_ranking PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(check_ranking sqlite3)

# Check weight shapes
add_executable(check_shapes
    check_shapes.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(check_shapes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(check_shapes sqlite3)

# Test token decode
add_executable(test_decode
    test_decode.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(test_decode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_decode sqlite3)

# Check vocab entries
add_executable(check_vocab_entries
    check_vocab_entries.cpp
    src/gguf_parser.cpp
    src/tensor.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(check_vocab_entries PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(check_vocab_entries sqlite3)

# Test encoding
add_executable(test_encoding
    test_encoding.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(test_encoding PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_encoding sqlite3)

# Test with Ollama tokens
add_executable(test_with_ollama_tokens
    test_with_ollama_tokens.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(test_with_ollama_tokens PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_with_ollama_tokens sqlite3)

# Check merges
add_executable(check_merges
    check_merges.cpp
    src/gguf_parser.cpp
    src/tensor.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(check_merges PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(check_merges sqlite3)

# Fast generation test
add_executable(test_gen_fast
    test_gen_fast.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(test_gen_fast PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_gen_fast sqlite3 ${OMP_LIBS})

# Test with reference token IDs (bypass tokenizer)
add_executable(test_reference_tokens
    test_reference_tokens.cpp
    src/inference.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(test_reference_tokens PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_reference_tokens sqlite3 ${OMP_LIBS})

# KV cache parity test
add_executable(test_kv_parity
    test_kv_parity.cpp
    src/inference.cpp
    src/inference_safetensors.cpp
    src/safetensors_parser.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/memory_lock.cpp
    src/logger.cpp
)
target_include_directories(test_kv_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_kv_parity sqlite3 ${OMP_LIBS})

# SafeTensors test executable
add_executable(test_safetensors
    src/test_safetensors.cpp
    src/logger.cpp
    src/tensor.cpp
    src/memory_lock.cpp
    src/safetensors_parser.cpp
)

target_include_directories(test_safetensors PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_safetensors PRIVATE Threads::Threads)

# Test inference with SafeTensors
add_executable(test_inference_safetensors
    src/test_inference_safetensors.cpp
    src/logger.cpp
    src/tensor.cpp
    src/matrix_ops.cpp
    src/memory_lock.cpp
    src/safetensors_parser.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/attention.cpp
    src/inference.cpp
    src/inference_safetensors.cpp
)

target_include_directories(test_inference_safetensors PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_inference_safetensors PRIVATE Threads::Threads sqlite3 ${OMP_LIBS})

# Set compile definitions for release builds
# target_compile_definitions(test_gen_fast PRIVATE NO_DEBUG_OUTPUT)  # Disabled - not currently used
