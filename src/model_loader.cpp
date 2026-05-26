#include "model_loader.h"
#include "gguf_parser.h"
#include "logger.h"
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <cstring>
#include <algorithm>

namespace fs = std::filesystem;

namespace ash {

// GGUF magic number
static const uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
static const uint32_t GGUF_VERSION = 3;

// SafeTensors magic
static const char* SAFETENSORS_MAGIC = "{\"__metadata__\"";

// =========================================================================
// ModelLoader Implementation
// =========================================================================

struct ModelLoader::Impl {
    ProgressCallback progress_callback;
};

ModelLoader::ModelLoader() : impl_(std::make_unique<Impl>()) {}
ModelLoader::~ModelLoader() = default;

std::unique_ptr<IModel> ModelLoader::load_local(
    const std::string& file_path,
    ModelFormat format
) {
    Logger::instance().info("Loading model from: " + file_path);
    
    // Check file exists
    if (!fs::exists(file_path)) {
        Logger::instance().error("Model file not found: " + file_path);
        return nullptr;
    }
    
    // Auto-detect format if needed
    if (format == ModelFormat::AUTO_DETECT) {
        format = detect_format(file_path);
        std::string format_name = (format == ModelFormat::GGUF ? "GGUF" : "Unknown");
        Logger::instance().info("Detected format: " + format_name);
    }
    
    // Create appropriate model instance
    std::unique_ptr<IModel> model;
    switch (format) {
        case ModelFormat::GGUF: {
            auto gguf = std::make_unique<GGUFModel>();
            if (gguf->load(file_path)) {
                model = std::move(gguf);
            }
            break;
        }
        case ModelFormat::SAFETENSORS: {
            auto st = std::make_unique<SafeTensorsModel>();
            if (st->load(file_path)) {
                model = std::move(st);
            }
            break;
        }
        default:
            Logger::instance().error("Unsupported model format");
            return nullptr;
    }
    
    if (model) {
        auto info = model->get_info();
        Logger::instance().info("✅ Model loaded: " + info.name + 
            " (" + std::to_string(info.file_size_bytes / (1024*1024)) + " MB)");
    }
    
    return model;
}

std::unique_ptr<IModel> ModelLoader::load_from_url(
    const std::string& url,
    const std::string& cache_dir,
    ModelFormat format
) {
    Logger::instance().info("Downloading model from: " + url);
    
    // Create cache directory
    fs::create_directories(cache_dir);
    
    // Extract filename from URL
    std::string filename = url.substr(url.find_last_of('/') + 1);
    std::string cache_path = cache_dir + "/" + filename;
    
    // Check if already cached
    if (fs::exists(cache_path)) {
        Logger::instance().info("Using cached model: " + cache_path);
        return load_local(cache_path, format);
    }
    
    // Download file
    bool success = ModelDownloader::download_file(url, cache_path, impl_->progress_callback);
    if (!success) {
        Logger::instance().error("Failed to download model");
        return nullptr;
    }
    
    // Load the downloaded model
    return load_local(cache_path, format);
}

std::unique_ptr<IModel> ModelLoader::load_from_huggingface(
    const std::string& repo_id,
    const std::string& filename,
    const std::string& cache_dir
) {
    Logger::instance().info("Loading from Hugging Face: " + repo_id);
    
    // TODO: Implement HF Hub API integration
    // For now, just log and return nullptr
    Logger::instance().warning("Hugging Face integration not yet implemented");
    Logger::instance().info("Manual download: https://huggingface.co/" + repo_id);
    
    return nullptr;
}

ModelFormat ModelLoader::detect_format(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        return ModelFormat::AUTO_DETECT;
    }
    
    // Read first 4 bytes for magic number
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    
    if (magic == GGUF_MAGIC) {
        return ModelFormat::GGUF;
    }
    
    // Check for SafeTensors JSON header
    file.seekg(0);
    char buffer[32];
    file.read(buffer, sizeof(buffer));
    if (std::strncmp(buffer, SAFETENSORS_MAGIC, std::strlen(SAFETENSORS_MAGIC)) == 0) {
        return ModelFormat::SAFETENSORS;
    }
    
    // Check file extension as fallback
    std::string ext = fs::path(file_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".gguf") return ModelFormat::GGUF;
    if (ext == ".safetensors") return ModelFormat::SAFETENSORS;
    if (ext == ".bin") return ModelFormat::PYTORCH_BIN;
    
    return ModelFormat::AUTO_DETECT;
}

QuantizationType ModelLoader::detect_quantization(const std::string& filename) {
    // Convert to lowercase for matching
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    // Check for quantization type in filename
    if (lower.find("q2_k") != std::string::npos) return QuantizationType::Q2_K;
    if (lower.find("q3_k_s") != std::string::npos) return QuantizationType::Q3_K_S;
    if (lower.find("q3_k_m") != std::string::npos) return QuantizationType::Q3_K_M;
    if (lower.find("q4_0") != std::string::npos) return QuantizationType::Q4_0;
    if (lower.find("q4_k_s") != std::string::npos) return QuantizationType::Q4_K_S;
    if (lower.find("q4_k_m") != std::string::npos) return QuantizationType::Q4_K_M;
    if (lower.find("q5_k_s") != std::string::npos) return QuantizationType::Q5_K_S;
    if (lower.find("q5_k_m") != std::string::npos) return QuantizationType::Q5_K_M;
    if (lower.find("q6_k") != std::string::npos) return QuantizationType::Q6_K;
    if (lower.find("q8_0") != std::string::npos) return QuantizationType::Q8_0;
    if (lower.find("f16") != std::string::npos) return QuantizationType::F16;
    if (lower.find("f32") != std::string::npos) return QuantizationType::F32;
    
    return QuantizationType::UNKNOWN;
}

void ModelLoader::set_progress_callback(ProgressCallback callback) {
    impl_->progress_callback = callback;
}

// =========================================================================
// GGUFModel Implementation
// =========================================================================

struct GGUFModel::Impl {
    bool loaded = false;
    ModelInfo info;
    void* mmap_ptr = nullptr;
    size_t mmap_size = 0;
};

GGUFModel::GGUFModel() : impl_(std::make_unique<Impl>()) {}
GGUFModel::~GGUFModel() {
    unload();
}

bool GGUFModel::load(const std::string& source_path) {
    Logger::instance().info("Loading GGUF model: " + source_path);
    
    std::ifstream file(source_path, std::ios::binary);
    if (!file) {
        Logger::instance().error("Failed to open GGUF file");
        return false;
    }
    
    // Read GGUF header
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    if (magic != GGUF_MAGIC) {
        Logger::instance().error("Invalid GGUF magic number");
        return false;
    }
    
    Logger::instance().info("GGUF version: " + std::to_string(version));
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0);
    
    // Fill in model info from GGUF metadata
    GGUFParser parser;
    if (parser.parse(source_path)) {
        impl_->info.architecture = parser.get_architecture();
        impl_->info.context_length = parser.get_context_length();
        impl_->info.embedding_dim = parser.get_embedding_dim();
        impl_->info.num_layers = parser.get_num_layers();
        impl_->info.num_heads = parser.get_num_heads();
        impl_->info.vocab_size = parser.get_vocab_size();
        
        Logger::instance().info("Metadata parsed: " + impl_->info.architecture + 
                               ", context=" + std::to_string(impl_->info.context_length));
    } else {
        // Fallback for failed parse
        impl_->info.architecture = "unknown";
        impl_->info.context_length = 2048;
    }

    impl_->info.name = fs::path(source_path).stem().string();
    impl_->info.version = std::to_string(version);
    impl_->info.format = ModelFormat::GGUF;
    impl_->info.quant_type = ModelLoader::detect_quantization(source_path);
    impl_->info.file_size_bytes = file_size;
    impl_->info.path = source_path;
    
    impl_->loaded = true;
    
    Logger::instance().info("GGUF model loaded successfully");
    Logger::instance().debug("  Size: " + std::to_string(file_size / (1024*1024)) + " MB");
    Logger::instance().debug("  Quant: " + std::to_string(static_cast<int>(impl_->info.quant_type)));
    
    return true;
}

bool GGUFModel::load_mmap(const std::string& file_path) {
    // TODO: Implement memory-mapped loading for large models
    // This is more efficient than reading entire file into RAM
    Logger::instance().warning("Memory-mapped loading not yet implemented");
    return load(file_path);
}

void GGUFModel::unload() {
    if (impl_->loaded) {
        Logger::instance().info("Unloading GGUF model: " + impl_->info.name);
        // TODO: Cleanup mmap if used
        impl_->loaded = false;
    }
}

bool GGUFModel::is_loaded() const {
    return impl_->loaded;
}

ModelInfo GGUFModel::get_info() const {
    return impl_->info;
}

size_t GGUFModel::get_memory_usage() const {
    // TODO: Track actual memory usage
    return impl_->loaded ? impl_->info.file_size_bytes : 0;
}

// =========================================================================
// SafeTensorsModel Implementation (stub)
// =========================================================================

struct SafeTensorsModel::Impl {
    bool loaded = false;
    ModelInfo info;
};

SafeTensorsModel::SafeTensorsModel() : impl_(std::make_unique<Impl>()) {}
SafeTensorsModel::~SafeTensorsModel() {
    unload();
}

bool SafeTensorsModel::load(const std::string& source_path) {
    Logger::instance().warning("SafeTensors format not yet fully implemented");
    return false;
}

void SafeTensorsModel::unload() {
    impl_->loaded = false;
}

bool SafeTensorsModel::is_loaded() const {
    return impl_->loaded;
}

ModelInfo SafeTensorsModel::get_info() const {
    return impl_->info;
}

size_t SafeTensorsModel::get_memory_usage() const {
    return 0;
}

// =========================================================================
// ModelRegistry Implementation (Enhanced for Ash-Forge)
// =========================================================================

struct ModelRegistry::Impl {
    std::unordered_map<std::string, std::unique_ptr<IModel>> models;
    std::unordered_map<std::string, ModelInfo> model_info;  // Metadata for all models
    std::unordered_map<std::string, time_t> last_used;       // LRU tracking
    std::recursive_mutex mutex;
};

ModelRegistry& ModelRegistry::instance() {
    static ModelRegistry instance;
    return instance;
}

ModelRegistry::ModelRegistry() : impl_(std::make_unique<Impl>()) {}
ModelRegistry::~ModelRegistry() {
    unload_all();
}

void ModelRegistry::register_model(const std::string& name, std::unique_ptr<IModel> model) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    impl_->models[name] = std::move(model);
    impl_->last_used[name] = std::time(nullptr);
    Logger::instance().info("Model registered: " + name);
}

IModel* ModelRegistry::get_model(const std::string& name) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    auto it = impl_->models.find(name);
    if (it != impl_->models.end()) {
        impl_->last_used[name] = std::time(nullptr);  // Update LRU
        return it->second.get();
    }
    return nullptr;
}

void ModelRegistry::unload_model(const std::string& name) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    impl_->models.erase(name);
    impl_->last_used.erase(name);
    Logger::instance().info("Model unloaded: " + name);
}

void ModelRegistry::unload_all() {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    Logger::instance().info("Unloading all models (" +
        std::to_string(impl_->models.size()) + ")");
    impl_->models.clear();
    impl_->last_used.clear();
}

std::vector<std::string> ModelRegistry::list_models() const {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    std::vector<std::string> names;
    for (const auto& [name, _] : impl_->models) {
        names.push_back(name);
    }
    return names;
}

size_t ModelRegistry::get_total_memory_usage() const {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    size_t total = 0;
    for (const auto& [_, model] : impl_->models) {
        total += model->get_memory_usage();
    }
    return total;
}

// ===== Ash-Forge Multi-Model Extensions =====

std::vector<ModelInfo> ModelRegistry::discover_models(const std::string& models_dir) {
    std::vector<ModelInfo> models;
    
    if (!fs::exists(models_dir)) {
        Logger::instance().warning("Models directory not found: " + models_dir);
        return models;
    }
    
    Logger::instance().info("Discovering models in: " + models_dir);
    
    for (auto& entry : fs::directory_iterator(models_dir)) {
        if (entry.path().extension() != ".gguf") continue;
        
        ModelInfo info;
        info.name = entry.path().stem().string();
        info.path = entry.path().string();
        info.file_size_bytes = fs::file_size(entry.path());
        info.format = ModelFormat::GGUF;
        
        // Parse model type from name
        std::string name_lower = info.name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        
        if (name_lower.find("core") != std::string::npos || 
            name_lower.find("router") != std::string::npos ||
            name_lower.find("scout") != std::string::npos) {
            info.type = ModelType::ROUTER;
            info.always_loaded = true;
            info.priority = 100;
            Logger::instance().info("  Found ROUTER: " + info.name);
        } 
        else if (name_lower.find("chat") != std::string::npos) {
            info.type = ModelType::PERSONALITY;
            info.always_loaded = false;
            info.priority = 80;
            Logger::instance().info("  Found PERSONALITY: " + info.name);
        } 
        else {
            info.type = ModelType::SPECIALIST;
            info.always_loaded = false;
            info.priority = 50;
            
            // Extract categories from name (e.g., "ash-python" → ["python"])
            if (name_lower.find("python") != std::string::npos) {
                info.categories = {"python", "pip", "django", "flask"};
            } else if (name_lower.find("javascript") != std::string::npos || 
                       name_lower.find("js") != std::string::npos) {
                info.categories = {"javascript", "js", "node", "npm", "react"};
            } else if (name_lower.find("debug") != std::string::npos) {
                info.categories = {"debugging", "error", "bug", "crash"};
            } else if (name_lower.find("system") != std::string::npos) {
                info.categories = {"systems", "linux", "docker", "kubernetes"};
            } else if (name_lower.find("creative") != std::string::npos) {
                info.categories = {"creative", "story", "poem", "writing"};
            }
            
            Logger::instance().info("  Found SPECIALIST: " + info.name + 
                " (categories: " + std::to_string(info.categories.size()) + ")");
        }
        
        // Store in registry
        impl_->model_info[info.name] = info;
        models.push_back(info);
    }
    
    Logger::instance().info("Discovered " + std::to_string(models.size()) + " models");
    return models;
}

bool ModelRegistry::load_model(const std::string& name, const std::string& file_path) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    
    // Check if already loaded
    if (impl_->models.count(name)) {
        Logger::instance().info("Model already loaded: " + name);
        impl_->last_used[name] = std::time(nullptr);
        return true;
    }
    
    // Check file exists
    if (!fs::exists(file_path)) {
        Logger::instance().error("Model file not found: " + file_path);
        return false;
    }
    
    // Use cached size from discovery if available (avoids opening the file a second time)
    size_t file_size = 0;
    if (impl_->model_info.count(name)) {
        file_size = impl_->model_info[name].file_size_bytes;
    }
    if (file_size == 0) {
        file_size = fs::file_size(file_path);
    }
    
    // Ensure space in budget
    if (!fits_in_budget(file_size)) {
        Logger::instance().warning("Model exceeds memory budget, attempting to free space...");
        if (!ensure_space(file_size)) {
            Logger::instance().error("Cannot free enough space for model: " + name);
            return false;
        }
    }
    
    // Create and load GGUF model
    auto model = std::make_unique<GGUFModel>();
    if (!model->load(file_path)) {
        Logger::instance().error("Failed to load model: " + name);
        return false;
    }
    
    // Register the model
    impl_->models[name] = std::move(model);
    impl_->last_used[name] = std::time(nullptr);
    
    // Store/update model info
    ModelInfo info = impl_->models[name]->get_info();
    if (impl_->model_info.count(name)) {
        // Preserve ash-forge metadata from discovery
        info.type = impl_->model_info[name].type;
        info.categories = impl_->model_info[name].categories;
        info.always_loaded = impl_->model_info[name].always_loaded;
        info.priority = impl_->model_info[name].priority;
    }
    impl_->model_info[name] = info;
    
    Logger::instance().info("Model loaded: " + name + 
        " (" + std::to_string(file_size / (1024*1024)) + " MB)");
    Logger::instance().info("  Total memory: " + 
        std::to_string(get_total_memory_usage() / (1024*1024)) + " MB / " +
        std::to_string(memory_budget_ / (1024*1024)) + " MB");
    
    return true;
}

bool ModelRegistry::is_loaded(const std::string& name) const {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    return impl_->models.count(name) > 0;
}

ModelInfo ModelRegistry::get_model_info(const std::string& name) const {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    auto it = impl_->model_info.find(name);
    if (it != impl_->model_info.end()) {
        return it->second;
    }
    return ModelInfo{};  // Empty info if not found
}

bool ModelRegistry::fits_in_budget(size_t additional_bytes) const {
    size_t current = get_total_memory_usage();
    return (current + additional_bytes) <= memory_budget_;
}

void ModelRegistry::mark_model_used(const std::string& name) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    impl_->last_used[name] = std::time(nullptr);
    
    // Increment query count if we have model info
    if (impl_->model_info.count(name)) {
        impl_->model_info[name].query_count++;
    }
}

std::string ModelRegistry::get_lru_model() const {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    
    if (impl_->last_used.empty()) {
        return "";
    }
    
    // Find least recently used model (that's not always_loaded)
    std::string lru_name;
    time_t oldest_time = std::time(nullptr);
    
    for (const auto& [name, last_time] : impl_->last_used) {
        // Skip always-loaded models (like ash-core)
        if (impl_->model_info.count(name) && impl_->model_info.at(name).always_loaded) {
            continue;
        }
        
        if (last_time < oldest_time) {
            oldest_time = last_time;
            lru_name = name;
        }
    }
    
    return lru_name;
}

bool ModelRegistry::ensure_space(size_t required_bytes) {
    // Evict LRU models until we have space
    while (!fits_in_budget(required_bytes)) {
        std::string lru = get_lru_model();
        if (lru.empty()) {
            Logger::instance().error("Cannot evict any more models (all are always_loaded)");
            return false;
        }
        
        Logger::instance().info("Evicting LRU model to free space: " + lru);
        unload_model(lru);
    }
    
    return true;
}

void ModelRegistry::cleanup_cold_models(int max_age_seconds) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    
    time_t now = std::time(nullptr);
    std::vector<std::string> to_unload;
    
    for (const auto& [name, last_time] : impl_->last_used) {
        // Skip always-loaded models
        if (impl_->model_info.count(name) && impl_->model_info.at(name).always_loaded) {
            continue;
        }
        
        int age = now - last_time;
        if (age > max_age_seconds) {
            to_unload.push_back(name);
        }
    }
    
    for (const auto& name : to_unload) {
        Logger::instance().info("Cleaning up cold model: " + name);
        impl_->models.erase(name);
        impl_->last_used.erase(name);
    }
    
    if (!to_unload.empty()) {
        Logger::instance().info("Cleaned up " + std::to_string(to_unload.size()) + " cold models");
    }
}

// =========================================================================
// ModelDownloader Implementation (stub)
// =========================================================================

bool ModelDownloader::download_file(
    const std::string& url,
    const std::string& output_path,
    ModelLoader::ProgressCallback progress
) {
    Logger::instance().warning("HTTP download not yet implemented");
    Logger::instance().info("Please download manually from: " + url);
    Logger::instance().info("Save to: " + output_path);
    return false;
}

bool ModelDownloader::download_from_huggingface(
    const std::string& repo_id,
    const std::string& filename,
    const std::string& output_path,
    const std::string& token,
    ModelLoader::ProgressCallback progress
) {
    Logger::instance().warning("Hugging Face API not yet implemented");
    std::string hf_url = "https://huggingface.co/" + repo_id;
    if (!filename.empty()) {
        hf_url += "/blob/main/" + filename;
    }
    Logger::instance().info("Please download manually from: " + hf_url);
    return false;
}

bool ModelDownloader::is_cached(const std::string& cache_dir, const std::string& filename) {
    return fs::exists(cache_dir + "/" + filename);
}

std::string ModelDownloader::get_cache_path(const std::string& cache_dir, const std::string& filename) {
    return cache_dir + "/" + filename;
}

} // namespace ash
