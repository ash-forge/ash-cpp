/**
 * SafeTensors model loading for InferenceEngine.
 * 
 * Implements load_model_from_safetensors and load_weights_from_safetensors_dir.
 * Link this file only in executables that need SafeTensors loading.
 */
#include "inference.h"
#include "safetensors_parser.h"
#include "tokenizer.h"
#include "attention.h"
#include "logger.h"
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <nlohmann/json.hpp>

namespace ash {

bool InferenceEngine::load_model_from_safetensors(
    const std::string& safetensors_dir,
    const std::string& gguf_tokenizer_path
) {
    namespace fs = std::filesystem;
    using json = nlohmann::json;

    // Parse config.json for model architecture
    std::string config_path = safetensors_dir + "/config.json";
    std::ifstream cfg_file(config_path);
    if (!cfg_file) {
        Logger::instance().error("config.json not found in: " + safetensors_dir);
        return false;
    }
    json cfg;
    try {
        cfg_file >> cfg;
    } catch (const json::exception& e) {
        Logger::instance().error("config.json parse error: " + std::string(e.what()));
        return false;
    }

    config_.architecture   = cfg.value("model_type", std::string("qwen2"));
    config_.vocab_size     = cfg.value("vocab_size", 151936);
    config_.hidden_dim     = cfg.value("hidden_size", 2048);
    config_.intermediate_dim = cfg.value("intermediate_size", 11008);
    config_.n_layers       = cfg.value("num_hidden_layers", 36);
    config_.n_heads        = cfg.value("num_attention_heads", 16);
    config_.n_kv_heads     = cfg.value("num_key_value_heads", 2);
    config_.max_seq_len    = cfg.value("max_position_embeddings", 32768);
    config_.rope_theta     = cfg.value("rope_theta", 1000000.0f);
    config_.rms_norm_eps   = cfg.value("rms_norm_eps", 1e-6f);
    config_.head_dim       = config_.hidden_dim / config_.n_heads;

    Logger::instance().info("SafeTensors config: arch=" + config_.architecture +
        " layers=" + std::to_string(config_.n_layers) +
        " hidden=" + std::to_string(config_.hidden_dim));

    if (!load_weights_from_safetensors_dir(safetensors_dir)) return false;

    tokenizer_ = TokenizerFactory::from_gguf(gguf_tokenizer_path);
    if (!tokenizer_) {
        Logger::instance().error("Failed to load tokenizer from: " + gguf_tokenizer_path);
        return false;
    }

    attention_ = std::make_unique<MultiHeadAttention>(config_.attention_config());
    weights_.loaded = true;
    return true;
}

bool InferenceEngine::load_weights_from_safetensors_dir(const std::string& dir) {
    namespace fs = std::filesystem;

    // Collect and sort shard paths
    std::vector<std::string> shard_paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".safetensors") {
            shard_paths.push_back(entry.path().string());
        }
    }
    std::sort(shard_paths.begin(), shard_paths.end());

    if (shard_paths.empty()) {
        Logger::instance().error("No .safetensors files in: " + dir);
        return false;
    }

    Logger::instance().info("Loading " + std::to_string(shard_paths.size()) + " shard(s)...");

    // Load all shards (each shard uses a single sequential file read via load_all_tensors_fast)
    std::unordered_map<std::string, Tensor> all_tensors;
    for (const auto& shard_path : shard_paths) {
        SafeTensorsParser parser;
        if (!parser.parse(shard_path)) {
            Logger::instance().error("Failed to parse: " + shard_path);
            return false;
        }
        auto shard_tensors = parser.load_all_tensors_fast();
        for (auto& [name, t] : shard_tensors) {
            all_tensors[name] = std::move(t);
        }
    }

    Logger::instance().info("Total tensors loaded: " + std::to_string(all_tensors.size()));

    // Move tensor out of map by HuggingFace name; returns empty Tensor if absent
    auto take = [&](const std::string& hf_name) -> Tensor {
        auto it = all_tensors.find(hf_name);
        if (it == all_tensors.end()) return Tensor();
        Tensor t = std::move(it->second);
        all_tensors.erase(it);
        return t;
    };

    // Embeddings
    weights_.token_embeddings = take("model.embed_tokens.weight");
    if (!weights_.token_embeddings.is_allocated()) {
        Logger::instance().error("Missing: model.embed_tokens.weight");
        return false;
    }

    // Final RMSNorm
    weights_.output_norm = take("model.norm.weight");
    if (!weights_.output_norm.is_allocated()) {
        Logger::instance().error("Missing: model.norm.weight");
        return false;
    }

    // LM head (may be tied to token embeddings)
    Tensor lm_head = take("lm_head.weight");
    weights_.output = lm_head.is_allocated()
        ? std::move(lm_head)
        : weights_.token_embeddings.clone();

    if (!lm_head.is_allocated()) {
        Logger::instance().info("lm_head.weight tied to embeddings");
    }

    // Layer weights: map HuggingFace naming → LayerWeights fields
    weights_.layers.resize(config_.n_layers);
    for (int i = 0; i < config_.n_layers; i++) {
        std::string p = "model.layers." + std::to_string(i) + ".";
        LayerWeights& lw = weights_.layers[i];

        lw.attn_norm = take(p + "input_layernorm.weight");
        lw.ffn_norm  = take(p + "post_attention_layernorm.weight");
        lw.wq        = take(p + "self_attn.q_proj.weight");
        lw.wk        = take(p + "self_attn.k_proj.weight");
        lw.wv        = take(p + "self_attn.v_proj.weight");
        lw.wo        = take(p + "self_attn.o_proj.weight");
        // Qwen2 has attention biases
        lw.bq        = take(p + "self_attn.q_proj.bias");
        lw.bk        = take(p + "self_attn.k_proj.bias");
        lw.bv        = take(p + "self_attn.v_proj.bias");
        // FFN (SwiGLU)
        lw.w_gate    = take(p + "mlp.gate_proj.weight");
        lw.w_up      = take(p + "mlp.up_proj.weight");
        lw.w_down    = take(p + "mlp.down_proj.weight");

        if (!lw.wq.is_allocated() || !lw.wk.is_allocated() || !lw.wv.is_allocated()) {
            Logger::instance().error("Missing attention weights at layer " + std::to_string(i));
            return false;
        }
    }

    Logger::instance().info("✅ SafeTensors weights loaded: " +
        std::to_string(config_.n_layers) + " layers, BF16→F32 converted");
    return true;
}

} // namespace ash
