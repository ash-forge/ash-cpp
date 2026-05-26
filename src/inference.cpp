#include "inference.h"
#include "matrix_ops.h"
#include "logger.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <chrono>

namespace ash {

InferenceEngine::InferenceEngine() {
    weights_.loaded = false;
}

InferenceEngine::~InferenceEngine() {
    // Cleanup handled by smart pointers
}

bool InferenceEngine::load_model(const std::string& gguf_path) {
    GGUFParser parser;
    if (!parser.parse(gguf_path)) {
        return false;
    }
    
    // Extract model config from parser
    config_.architecture = parser.get_architecture();
    if (config_.architecture.empty()) {
        Logger::instance().error("GGUF: general.architecture missing — cannot load model");
        return false;
    }
    config_.vocab_size = parser.get_vocab_size();
    config_.hidden_dim = parser.get_embedding_dim();
    config_.n_layers = parser.get_num_layers();
    config_.n_heads = parser.get_num_heads();
    config_.n_kv_heads = parser.get_num_kv_heads();
    config_.max_seq_len = parser.get_context_length();
    
    // Validate critical dimensions before proceeding
    if (config_.hidden_dim == 0 || config_.n_layers == 0 || config_.n_heads == 0) {
        Logger::instance().error("GGUF: critical model dimensions are zero — metadata may be malformed");
        Logger::instance().error("  hidden_dim=" + std::to_string(config_.hidden_dim) +
                                 " n_layers=" + std::to_string(config_.n_layers) +
                                 " n_heads=" + std::to_string(config_.n_heads));
        return false;
    }
    
    std::string prefix = config_.architecture + ".";
    config_.intermediate_dim = static_cast<int>(parser.get_uint(prefix + "feed_forward_length", 0));
    if (config_.intermediate_dim == 0) {
        // Fallback: SwiGLU FFN is typically ~8/3 × hidden_dim (rounded up to multiple of 256)
        config_.intermediate_dim = ((config_.hidden_dim * 8 / 3) + 255) & ~255;
        Logger::instance().warning("GGUF: feed_forward_length missing, estimating as " +
                                   std::to_string(config_.intermediate_dim));
    }
    
    // rope_theta: check arch-specific key, then apply sensible architecture defaults
    config_.rope_theta = parser.get_float(prefix + "rope.freq_base", 0.0f);
    if (config_.rope_theta == 0.0f) {
        if (config_.architecture == "qwen2" || config_.architecture == "qwen3") {
            config_.rope_theta = 1000000.0f;
        } else if (config_.architecture == "llama" || config_.architecture == "mistral") {
            config_.rope_theta = 500000.0f;  // Llama3+ default
        } else {
            config_.rope_theta = 10000.0f;   // Classic RoPE default (Llama2, Gemma)
        }
        Logger::instance().warning("GGUF: rope.freq_base missing, using default " +
                                   std::to_string(config_.rope_theta));
    }
    
    config_.rms_norm_eps = parser.get_float(prefix + "attention.layer_norm_rms_epsilon", 0.0f);
    if (config_.rms_norm_eps == 0.0f) {
        config_.rms_norm_eps = 1e-5f;  // Llama/Mistral use 1e-5; Gemma uses 1e-6 (re-specified in file)
    }
    
    // head_dim: use explicit value if provided (Gemma, Phi-3 set attention.key_length)
    uint64_t explicit_head_dim = parser.get_uint(prefix + "attention.key_length", 0);
    config_.head_dim = explicit_head_dim > 0
        ? static_cast<int>(explicit_head_dim)
        : config_.hidden_dim / config_.n_heads;
    
    // Debug config values
    Logger::instance().info("Model config loaded:");
    Logger::instance().info("  arch=" + config_.architecture);
    Logger::instance().info("  n_layers=" + std::to_string(config_.n_layers));
    Logger::instance().info("  n_heads=" + std::to_string(config_.n_heads));
    Logger::instance().info("  n_kv_heads=" + std::to_string(config_.n_kv_heads));
    Logger::instance().info("  hidden_dim=" + std::to_string(config_.hidden_dim));
    Logger::instance().info("  intermediate_dim=" + std::to_string(config_.intermediate_dim));
    Logger::instance().info("  head_dim=" + std::to_string(config_.head_dim));
    Logger::instance().info("  max_seq_len=" + std::to_string(config_.max_seq_len));
    Logger::instance().info("  vocab_size=" + std::to_string(config_.vocab_size));
    Logger::instance().info("  rope_theta=" + std::to_string(config_.rope_theta));
    Logger::instance().info("  rms_norm_eps=" + std::to_string(config_.rms_norm_eps));
    
    // Load weights
    if (!load_weights_from_gguf(parser)) {
        return false;
    }
    
    // Create tokenizer from GGUF vocab
    tokenizer_ = TokenizerFactory::from_gguf(gguf_path);
    if (!tokenizer_) {
        return false;
    }
    
    // Create attention module
    attention_ = std::make_unique<MultiHeadAttention>(config_.attention_config());
    
    weights_.loaded = true;
    return true;
}

bool InferenceEngine::load_weights_from_gguf(GGUFParser& parser) {
    // Load token embeddings
    weights_.token_embeddings = parser.load_tensor("token_embd.weight");
    if (!weights_.token_embeddings.is_allocated()) {
        return false;
    }
    
    // Load layer weights  
    std::string prefix = "blk.";  // Tensor names are just "blk.X.Y", not "arch.blk.X.Y"
    weights_.layers.resize(config_.n_layers);
    
    for (size_t i = 0; i < config_.n_layers; i++) {
        std::string layer_prefix = prefix + std::to_string(i) + ".";
        LayerWeights& layer = weights_.layers[i];
        
        // Attention weights
        layer.wq = parser.load_tensor(layer_prefix + "attn_q.weight");
        layer.wk = parser.load_tensor(layer_prefix + "attn_k.weight");
        layer.wv = parser.load_tensor(layer_prefix + "attn_v.weight");
        layer.wo = parser.load_tensor(layer_prefix + "attn_output.weight");
        
        // Attention biases (Qwen2 has these, optional for other models)
        if (parser.find_tensor(layer_prefix + "attn_q.bias")) {
            layer.bq = parser.load_tensor(layer_prefix + "attn_q.bias");
        }
        if (parser.find_tensor(layer_prefix + "attn_k.bias")) {
            layer.bk = parser.load_tensor(layer_prefix + "attn_k.bias");
        }
        if (parser.find_tensor(layer_prefix + "attn_v.bias")) {
            layer.bv = parser.load_tensor(layer_prefix + "attn_v.bias");
        }
        
        // FFN weights
        layer.w_gate = parser.load_tensor(layer_prefix + "ffn_gate.weight");
        layer.w_up = parser.load_tensor(layer_prefix + "ffn_up.weight");
        layer.w_down = parser.load_tensor(layer_prefix + "ffn_down.weight");
        
        // Layer norms
        layer.attn_norm = parser.load_tensor(layer_prefix + "attn_norm.weight");
        layer.ffn_norm = parser.load_tensor(layer_prefix + "ffn_norm.weight");
        
        // Verify critical weights loaded
        if (!layer.wq.is_allocated() || !layer.wk.is_allocated() || !layer.wv.is_allocated()) {
            return false;
        }
    }
    
    // Output norm and projection
    weights_.output_norm = parser.load_tensor("output_norm.weight");
    
    // Try to load output projection (may not exist - will use tied embeddings)
    if (parser.find_tensor("output.weight")) {
        weights_.output = parser.load_tensor("output.weight");
    }
    
    // Often output weight is tied to embeddings
    if (!weights_.output.is_allocated()) {
        weights_.output = weights_.token_embeddings.clone();
    }
    
    // CRITICAL OPTIMIZATION: Dequantize all weights once at load time
    // This is much faster than dequantizing on every forward pass
    std::cout << "🔥 Dequantizing weights..." << std::flush;
    auto deq_start = std::chrono::high_resolution_clock::now();
    
    size_t deq_count = 0;
    
    // Dequantize embeddings and output
    if (weights_.token_embeddings.dtype() != DType::F32) {
        weights_.token_embeddings = weights_.token_embeddings.dequantize();
        deq_count++;
    }
    if (weights_.output.dtype() != DType::F32) {
        weights_.output = weights_.output.dequantize();
        deq_count++;
    }
    
    // Dequantize all layer weights
    for (size_t i = 0; i < config_.n_layers; i++) {
        LayerWeights& layer = weights_.layers[i];
        
        // Attention weights
        if (layer.wq.dtype() != DType::F32) { layer.wq = layer.wq.dequantize(); deq_count++; }
        if (layer.wk.dtype() != DType::F32) { layer.wk = layer.wk.dequantize(); deq_count++; }
        if (layer.wv.dtype() != DType::F32) { layer.wv = layer.wv.dequantize(); deq_count++; }
        if (layer.wo.dtype() != DType::F32) { layer.wo = layer.wo.dequantize(); deq_count++; }
        
        // Attention biases
        if (layer.bq.is_allocated() && layer.bq.dtype() != DType::F32) { layer.bq = layer.bq.dequantize(); deq_count++; }
        if (layer.bk.is_allocated() && layer.bk.dtype() != DType::F32) { layer.bk = layer.bk.dequantize(); deq_count++; }
        if (layer.bv.is_allocated() && layer.bv.dtype() != DType::F32) { layer.bv = layer.bv.dequantize(); deq_count++; }
        
        // FFN weights
        if (layer.w_gate.dtype() != DType::F32) { layer.w_gate = layer.w_gate.dequantize(); deq_count++; }
        if (layer.w_up.dtype() != DType::F32) { layer.w_up = layer.w_up.dequantize(); deq_count++; }
        if (layer.w_down.dtype() != DType::F32) { layer.w_down = layer.w_down.dequantize(); deq_count++; }
        
        // Norms are already F32
    }
    
    auto deq_end = std::chrono::high_resolution_clock::now();
    auto deq_ms = std::chrono::duration_cast<std::chrono::milliseconds>(deq_end - deq_start).count();
    std::cout << " " << deq_count << " tensors in " << deq_ms << "ms\n";
    
    return true;
}

Tensor InferenceEngine::forward(const std::vector<TokenID>& tokens, KVCache* kv_cache, int start_pos) {
    if (!weights_.loaded) {
        return Tensor();
    }
    
    int seq_len = tokens.size();
    
    // Get token embeddings: [seq_len, hidden_dim]
    Tensor x = Tensor::empty({seq_len, config_.hidden_dim}, DType::F32);
    
    bool use_emb_scaling = (config_.architecture == "gemma");
    float emb_scale = use_emb_scaling ? std::sqrt(static_cast<float>(config_.hidden_dim)) : 1.0f;
    
    for (int i = 0; i < seq_len; i++) {
        const float* emb_data = weights_.token_embeddings.data_f32() + tokens[i] * config_.hidden_dim;
        float* x_data = x.data_f32() + i * config_.hidden_dim;
        if (use_emb_scaling) {
            for (int d = 0; d < config_.hidden_dim; d++) {
                x_data[d] = emb_data[d] * emb_scale;
            }
        } else {
            std::memcpy(x_data, emb_data, config_.hidden_dim * sizeof(float));
        }
    }
    
    // Forward through all layers
    for (int layer_idx = 0; layer_idx < (int)config_.n_layers; layer_idx++) {
        x = forward_layer(x, weights_.layers[layer_idx], layer_idx, start_pos, kv_cache);
    }
    
    // Final RMSNorm + output projection
    x = rmsnorm(x, weights_.output_norm, config_.rms_norm_eps);
    Tensor logits = matmul_transposed(x, weights_.output, false, true);
    
    return logits;
}

Tensor InferenceEngine::forward_layer(
    const Tensor& x,
    const LayerWeights& layer,
    int layer_idx,
    int pos,
    KVCache* kv_cache
) {
    // Pre-attention RMSNorm + attention + residual
    Tensor x_norm = rmsnorm(x, layer.attn_norm, config_.rms_norm_eps);
    Tensor attn_out = attention_->forward(
        x_norm,
        layer.wq, layer.wk, layer.wv, layer.wo,
        layer.bq, layer.bk, layer.bv,
        layer_idx, pos, kv_cache
    );
    Tensor x_attn = add(x, attn_out);
    
    // Pre-FFN RMSNorm + FFN + residual
    Tensor x_attn_norm = rmsnorm(x_attn, layer.ffn_norm, config_.rms_norm_eps);
    Tensor ffn_out = forward_ffn(x_attn_norm, layer);
    
    return add(x_attn, ffn_out);
}

Tensor InferenceEngine::forward_ffn(const Tensor& x, const LayerWeights& layer) {
    // SwiGLU: silu(gate(x)) * up(x) @ down
    // Weights are stored as [out_features, in_features] so we need to transpose
    Tensor gate = matmul_transposed(x, layer.w_gate, false, true);  // x @ w_gate^T
    Tensor up = matmul_transposed(x, layer.w_up, false, true);      // x @ w_up^T
    
    // Apply SiLU to gate
    gate = silu(gate);
    
    // Element-wise multiply
    Tensor gated = multiply(gate, up);
    
    // Down projection
    Tensor out = matmul_transposed(gated, layer.w_down, false, true);  // gated @ w_down^T
    
    return out;
}

GenerationResult InferenceEngine::generate(
    const std::string& prompt,
    const SamplingConfig& sampling
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    GenerationResult result;
    result.tokens_generated = 0;
    result.stopped_early = false;
    
    if (!weights_.loaded) {
        result.stop_reason = "model_not_loaded";
        return result;
    }
    
    // Tokenize prompt
    std::vector<TokenID> tokens = tokenizer_->encode(prompt);
    if (tokens.empty()) {
        result.stop_reason = "tokenization_failed";
        return result;
    }
    const size_t prompt_len = tokens.size();
    
    // Build effective stop token set: caller's list + model EOS + <|im_end|> (151645)
    SamplingConfig effective_sampling = sampling;
    auto& stop = effective_sampling.stop_tokens;
    auto add_stop = [&](TokenID t) {
        if (t != 0 && std::find(stop.begin(), stop.end(), t) == stop.end())
            stop.push_back(t);
    };
    add_stop(tokenizer_->special_tokens().eos_token);
    add_stop(151645);  // <|im_end|> — Qwen2.5 chat end-of-turn token
    
    // Create KV cache for this generation
    // Constructor: KVCache(max_seq_len, n_layers, n_kv_heads, head_dim)
    KVCache kv_cache(config_.max_seq_len, config_.n_layers, config_.n_kv_heads, config_.head_dim);
    
    // PREFILL PHASE: Process entire prompt to fill KV cache
    Tensor logits = forward(tokens, &kv_cache, 0);  // start_pos = 0
    
    // Get first generated token from last position
    int seq_len = logits.shape().dims[0];
    int vocab_size = logits.shape().dims[1];
    Tensor last_logits = Tensor::empty({vocab_size}, DType::F32);
    const float* logits_data = logits.data_f32() + (seq_len - 1) * vocab_size;
    std::memcpy(last_logits.data_f32(), logits_data, vocab_size * sizeof(float));
    
    inference_utils::apply_repetition_penalty(last_logits, tokens,
        effective_sampling.repetition_penalty, effective_sampling.repetition_penalty_range);
    TokenID next_token = sample_token(last_logits, effective_sampling);
    
    std::cout << "." << std::flush;
    
    if (inference_utils::is_stop_token(next_token, effective_sampling.stop_tokens)) {
        result.stopped_early = true;
        result.stop_reason = "stop_token";
        result.text = tokenizer_->decode({tokens.begin() + prompt_len, tokens.end()}, true);
        result.tokens = tokens;
        auto end_time = std::chrono::high_resolution_clock::now();
        result.generation_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        return result;
    }
    
    tokens.push_back(next_token);
    result.tokens_generated++;
    Logger::instance().debug("Generated token[0]: " + std::to_string(next_token) + " = '" + tokenizer_->decode({next_token}) + "'");
    
    // GENERATION PHASE: Generate one token at a time
    for (int i = 1; i < effective_sampling.max_tokens; i++) {
        next_token = generate_next_token(tokens, effective_sampling, &kv_cache);
        
        // Simple progress indicator
        std::cout << "." << std::flush;
        
        // Check stop conditions
        if (inference_utils::is_stop_token(next_token, effective_sampling.stop_tokens)) {
            result.stopped_early = true;
            result.stop_reason = "stop_token";
            break;
        }
        
        tokens.push_back(next_token);
        result.tokens_generated++;
        Logger::instance().debug("Generated token[" + std::to_string(i) + "]: " + std::to_string(next_token) + " = '" + tokenizer_->decode({next_token}) + "'");
        
        // Check max length
        if (tokens.size() >= (size_t)config_.max_seq_len) {
            result.stopped_early = true;
            result.stop_reason = "max_length";
            break;
        }
    }
    std::cout << "\n";
    
    // Decode only generated tokens (skip prompt + special tokens like <|im_start|>)
    result.text = tokenizer_->decode({tokens.begin() + prompt_len, tokens.end()}, true);
    
    // Strip common training artifacts that some fine-tuned models emit at the start
    for (const auto& prefix : {"user\n", "assistant\n"}) {
        if (result.text.substr(0, std::strlen(prefix)) == prefix) {
            result.text = result.text.substr(std::strlen(prefix));
        }
    }
    // Strip any remaining leading whitespace/newlines
    size_t text_start = result.text.find_first_not_of(" \n\r\t");
    if (text_start != std::string::npos && text_start > 0) {
        result.text = result.text.substr(text_start);
    }
    result.tokens = tokens;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.generation_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    if (!result.stopped_early) {
        result.stop_reason = "max_tokens";
    }
    
    return result;
}

GenerationResult InferenceEngine::generate_from_tokens(
    const std::vector<TokenID>& prompt_tokens,
    const SamplingConfig& sampling
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    GenerationResult result;
    result.tokens_generated = 0;
    result.stopped_early = false;
    
    if (!weights_.loaded) {
        result.stop_reason = "model_not_loaded";
        return result;
    }
    
    if (prompt_tokens.empty()) {
        result.stop_reason = "empty_prompt";
        return result;
    }
    
    // Start with the provided tokens
    std::vector<TokenID> tokens = prompt_tokens;
    
    // Create KV cache
    KVCache kv_cache(config_.max_seq_len, config_.n_layers, config_.n_kv_heads, config_.head_dim);
    
    // PREFILL PHASE: Process entire prompt to fill KV cache
    Tensor logits = forward(tokens, &kv_cache, 0);
    
    // Get first generated token from last position
    int seq_len = logits.shape().dims[0];
    int vocab_size = logits.shape().dims[1];
    Tensor last_logits = Tensor::empty({vocab_size}, DType::F32);
    const float* logits_data = logits.data_f32() + (seq_len - 1) * vocab_size;
    std::memcpy(last_logits.data_f32(), logits_data, vocab_size * sizeof(float));
    TokenID next_token = sample_token(last_logits, sampling);
    
    std::cout << "." << std::flush;
    
    if (inference_utils::is_stop_token(next_token, sampling.stop_tokens)) {
        result.stopped_early = true;
        result.stop_reason = "stop_token";
        result.text = tokenizer_->decode(tokens);
        result.tokens = tokens;
        auto end_time = std::chrono::high_resolution_clock::now();
        result.generation_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        return result;
    }
    
    tokens.push_back(next_token);
    result.tokens_generated++;
    
    // GENERATION PHASE: Generate one token at a time
    for (int i = 1; i < sampling.max_tokens; i++) {
        next_token = generate_next_token(tokens, sampling, &kv_cache);
        
        std::cout << "." << std::flush;
        
        if (inference_utils::is_stop_token(next_token, sampling.stop_tokens)) {
            result.stopped_early = true;
            result.stop_reason = "stop_token";
            break;
        }
        
        tokens.push_back(next_token);
        result.tokens_generated++;
        
        if (tokens.size() >= (size_t)config_.max_seq_len) {
            result.stopped_early = true;
            result.stop_reason = "max_length";
            break;
        }
    }
    std::cout << "\n";
    
    // Decode tokens to text
    result.text = tokenizer_->decode(tokens);
    result.tokens = tokens;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.generation_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    if (!result.stopped_early) {
        result.stop_reason = "max_tokens";
    }
    
    return result;
}

TokenID InferenceEngine::generate_next_token(
    const std::vector<TokenID>& context,
    const SamplingConfig& sampling,
    KVCache* kv_cache
) {
    // CRITICAL: Only pass the LAST token for generation!
    // KV cache already has all previous tokens' keys/values
    // Processing all tokens would be O(n^2) complexity
    std::vector<TokenID> new_token_vec = {context.back()};
    int start_pos = context.size() - 1;  // Position of the last token (the one we're processing)
    
    Logger::instance().debug("generate_next_token: pos=" + std::to_string(start_pos) + 
                            ", token=" + std::to_string(new_token_vec[0]) + 
                            ", kv_cache_len=" + std::to_string(kv_cache ? kv_cache->seq_len() : -1));
    
    // Forward pass with just the new token at correct position
    Tensor logits = forward(new_token_vec, kv_cache, start_pos);
    
    // Get logits: [1, vocab_size]
    int vocab_size = logits.shape().dims[1];
    
    Tensor last_logits = Tensor::empty({vocab_size}, DType::F32);
    const float* logits_data = logits.data_f32();
    std::memcpy(last_logits.data_f32(), logits_data, vocab_size * sizeof(float));
    
    // Apply repetition penalty before sampling
    inference_utils::apply_repetition_penalty(last_logits, context,
        sampling.repetition_penalty, sampling.repetition_penalty_range);
    
    // Sample token
    return sample_token(last_logits, sampling);
}

TokenID InferenceEngine::sample_token(const Tensor& logits, const SamplingConfig& config) {
    if (!config.use_sampling || config.temperature == 0.0f) {
        return sample_greedy(logits);
    }
    
    // Use top-k or top-p sampling
    if (config.top_k > 0) {
        return sample_top_k(logits, config.top_k, config.temperature);
    } else if (config.top_p > 0.0f && config.top_p < 1.0f) {
        return sample_top_p(logits, config.top_p, config.temperature);
    }
    
    // Fallback to greedy
    return sample_greedy(logits);
}

TokenID InferenceEngine::sample_greedy(const Tensor& logits) {
    const float* data = logits.data_f32();
    int vocab_size = logits.num_elements();
    
    int max_idx = 0;
    float max_val = data[0];
    
    for (int i = 1; i < vocab_size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

TokenID InferenceEngine::sample_top_k(const Tensor& logits, int k, float temperature) {
    // Get top-k indices
    auto top_k = inference_utils::top_k_indices(logits, k);
    
    // Extract top-k logits
    Tensor top_k_logits({k}, DType::F32);
    const float* logits_data = logits.data_f32();
    float* top_k_data = top_k_logits.data_f32();
    
    for (int i = 0; i < k; i++) {
        top_k_data[i] = logits_data[top_k[i]];
    }
    
    // Apply temperature and softmax
    Tensor probs = inference_utils::softmax_temperature(top_k_logits, temperature);
    
    // Sample from distribution
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    float r = dis(gen);
    float cumsum = 0.0f;
    const float* probs_data = probs.data_f32();
    
    for (int i = 0; i < k; i++) {
        cumsum += probs_data[i];
        if (r < cumsum) {
            return top_k[i];
        }
    }
    
    // Fallback to last token
    return top_k[k - 1];
}

TokenID InferenceEngine::sample_top_p(const Tensor& logits, float p, float temperature) {
    int vocab_size = logits.num_elements();
    
    // Apply temperature
    Tensor temp_logits = logits.clone();
    inference_utils::apply_temperature(temp_logits, temperature);
    
    // Compute softmax
    Tensor probs = softmax(temp_logits);
    
    // Sort indices by probability (descending)
    std::vector<std::pair<float, int>> prob_idx;
    const float* probs_data = probs.data_f32();
    for (int i = 0; i < vocab_size; i++) {
        prob_idx.push_back({probs_data[i], i});
    }
    std::sort(prob_idx.begin(), prob_idx.end(), std::greater<>());
    
    // Find nucleus (cumulative probability > p)
    float cumsum = 0.0f;
    int nucleus_size = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += prob_idx[i].first;
        nucleus_size++;
        if (cumsum >= p) break;
    }
    
    // Sample from nucleus
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, cumsum);
    
    float r = dis(gen);
    float sum = 0.0f;
    for (int i = 0; i < nucleus_size; i++) {
        sum += prob_idx[i].first;
        if (r < sum) {
            return prob_idx[i].second;
        }
    }
    
    // Fallback to most likely
    return prob_idx[0].second;
}

// Inference utilities implementation
namespace inference_utils {

void apply_temperature(Tensor& logits, float temperature) {
    if (temperature == 1.0f) return;
    
    float* data = logits.data_f32();
    int size = logits.num_elements();
    
    for (int i = 0; i < size; i++) {
        data[i] /= temperature;
    }
}

void apply_repetition_penalty(Tensor& logits, const std::vector<TokenID>& context,
                              float penalty, int range) {
    if (penalty == 1.0f || context.empty()) return;
    float* data = logits.data_f32();
    int vocab_size = logits.num_elements();
    
    int start = std::max(0, static_cast<int>(context.size()) - range);
    for (int i = start; i < static_cast<int>(context.size()); i++) {
        TokenID token = context[i];
        if (token < 0 || token >= vocab_size) continue;
        // Standard repetition penalty: divide positive logits, multiply negative
        if (data[token] > 0.0f)
            data[token] /= penalty;
        else
            data[token] *= penalty;
    }
}

std::vector<int> top_k_indices(const Tensor& logits, int k) {
    int vocab_size = logits.num_elements();
    k = std::min(k, vocab_size);
    
    // Create pairs of (logit, index)
    std::vector<std::pair<float, int>> logit_idx;
    const float* data = logits.data_f32();
    for (int i = 0; i < vocab_size; i++) {
        logit_idx.push_back({data[i], i});
    }
    
    // Partial sort to get top-k
    std::partial_sort(logit_idx.begin(), logit_idx.begin() + k, logit_idx.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Extract indices
    std::vector<int> indices(k);
    for (int i = 0; i < k; i++) {
        indices[i] = logit_idx[i].second;
    }
    
    return indices;
}

Tensor softmax_temperature(const Tensor& logits, float temperature) {
    Tensor temp_logits = logits.clone();
    apply_temperature(temp_logits, temperature);
    return softmax(temp_logits);
}

bool is_stop_token(TokenID token, const std::vector<TokenID>& stop_tokens) {
    return std::find(stop_tokens.begin(), stop_tokens.end(), token) != stop_tokens.end();
}

} // namespace inference_utils

} // namespace ash
