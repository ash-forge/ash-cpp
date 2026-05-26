#include "gguf_parser.h"
#include "logger.h"
#include <fstream>
#include <cstring>

namespace ash {

// GGUF constants
static const uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"

// Convert GGUF type to DType
DType gguf_type_to_dtype(GGUFTensorType gguf_type) {
    switch (gguf_type) {
        case GGUFTensorType::F32: return DType::F32;
        case GGUFTensorType::F16: return DType::F16;
        case GGUFTensorType::Q4_0: return DType::Q4_0;
        case GGUFTensorType::Q8_0: return DType::Q8_0;
        case GGUFTensorType::Q4_K: return DType::Q4_K;
        case GGUFTensorType::Q5_K: return DType::Q5_K;
        case GGUFTensorType::Q6_K: return DType::Q6_K;
        case GGUFTensorType::I8: return DType::I8;
        case GGUFTensorType::I16: return DType::I16;
        case GGUFTensorType::I32: return DType::I32;
        default:
            Logger::instance().warning("Unknown GGUF type, defaulting to F32");
            return DType::F32;
    }
}

GGUFParser::GGUFParser() = default;
GGUFParser::~GGUFParser() = default;

// Maximum array elements to store — anything larger (e.g. vocabulary lists) is skipped
static constexpr uint64_t MAX_STORED_ARRAY = 4096;

// Returns element byte width for fixed-width GGUF types, 0 for variable-width
static size_t fixed_element_size(GGUFMetadataType type) {
    switch (type) {
        case GGUFMetadataType::UINT8:  case GGUFMetadataType::INT8:  case GGUFMetadataType::BOOL:   return 1;
        case GGUFMetadataType::UINT16: case GGUFMetadataType::INT16:                                 return 2;
        case GGUFMetadataType::UINT32: case GGUFMetadataType::INT32: case GGUFMetadataType::FLOAT32: return 4;
        case GGUFMetadataType::UINT64: case GGUFMetadataType::INT64: case GGUFMetadataType::FLOAT64: return 8;
        default: return 0; // STRING or ARRAY — variable width
    }
}

// Advance file position past one metadata value without storing it
static void skip_bytes(std::ifstream& file, uint64_t n) {
    static char discard[65536];
    while (n > 0) {
        size_t chunk = (n < sizeof(discard)) ? (size_t)n : sizeof(discard);
        file.read(discard, chunk);
        n -= chunk;
    }
}

static void skip_metadata_value(std::ifstream& file, GGUFMetadataType type) {
    size_t fixed = fixed_element_size(type);
    if (fixed > 0) {
        skip_bytes(file, fixed);
        return;
    }
    switch (type) {
        case GGUFMetadataType::STRING: {
            uint64_t len;
            file.read(reinterpret_cast<char*>(&len), 8);
            if (len < 8 * 1024 * 1024) skip_bytes(file, len);
            break;
        }
        case GGUFMetadataType::ARRAY: {
            uint32_t arr_type; uint64_t arr_len;
            file.read(reinterpret_cast<char*>(&arr_type), 4);
            file.read(reinterpret_cast<char*>(&arr_len), 8);
            size_t elem_size = fixed_element_size(static_cast<GGUFMetadataType>(arr_type));
            if (elem_size > 0) {
                skip_bytes(file, arr_len * elem_size); // Single sequential read for fixed-width arrays
            } else {
                for (uint64_t i = 0; i < arr_len; ++i)
                    skip_metadata_value(file, static_cast<GGUFMetadataType>(arr_type));
            }
            break;
        }
        default: break;
    }
}

std::string GGUFParser::read_string(std::ifstream& file) {
    uint64_t len;
    file.read(reinterpret_cast<char*>(&len), sizeof(len));
    
    if (len == 0 || len > 1024*1024) { // Sanity check
        return "";
    }
    
    std::string str(len, '\0');
    file.read(&str[0], len);
    return str;
}

bool GGUFParser::read_metadata_value(std::ifstream& file, GGUFMetadataType type, GGUFMetadataValue& out) {
    out.type = type;
    
    switch (type) {
        case GGUFMetadataType::UINT8: {
            uint8_t val;
            file.read(reinterpret_cast<char*>(&val), 1);
            out.uint_value = val;
            break;
        }
        case GGUFMetadataType::INT8: {
            int8_t val;
            file.read(reinterpret_cast<char*>(&val), 1);
            out.int_value = val;
            break;
        }
        case GGUFMetadataType::UINT16: {
            uint16_t val;
            file.read(reinterpret_cast<char*>(&val), 2);
            out.uint_value = val;
            break;
        }
        case GGUFMetadataType::INT16: {
            int16_t val;
            file.read(reinterpret_cast<char*>(&val), 2);
            out.int_value = val;
            break;
        }
        case GGUFMetadataType::UINT32: {
            uint32_t val;
            file.read(reinterpret_cast<char*>(&val), 4);
            out.uint_value = val;
            break;
        }
        case GGUFMetadataType::INT32: {
            int32_t val;
            file.read(reinterpret_cast<char*>(&val), 4);
            out.int_value = val;
            break;
        }
        case GGUFMetadataType::UINT64: {
            file.read(reinterpret_cast<char*>(&out.uint_value), 8);
            break;
        }
        case GGUFMetadataType::INT64: {
            file.read(reinterpret_cast<char*>(&out.int_value), 8);
            break;
        }
        case GGUFMetadataType::FLOAT32: {
            float val;
            file.read(reinterpret_cast<char*>(&val), 4);
            out.float_value = val;
            break;
        }
        case GGUFMetadataType::FLOAT64: {
            file.read(reinterpret_cast<char*>(&out.float_value), 8);
            break;
        }
        case GGUFMetadataType::BOOL: {
            uint8_t val;
            file.read(reinterpret_cast<char*>(&val), 1);
            out.bool_value = (val != 0);
            break;
        }
        case GGUFMetadataType::STRING: {
            out.string_value = read_string(file);
            break;
        }
        case GGUFMetadataType::ARRAY: {
            uint32_t array_type;
            uint64_t array_len;
            file.read(reinterpret_cast<char*>(&array_type), 4);
            file.read(reinterpret_cast<char*>(&array_len), 8);

            if (array_len > MAX_STORED_ARRAY) {
                // Skip large arrays (e.g. 152K-entry tokenizer vocab) efficiently
                size_t elem_size = fixed_element_size(static_cast<GGUFMetadataType>(array_type));
                if (elem_size > 0) {
                    file.seekg(array_len * elem_size, std::ios::cur);
                } else {
                    for (uint64_t i = 0; i < array_len; ++i)
                        skip_metadata_value(file, static_cast<GGUFMetadataType>(array_type));
                }
                out.array_value.clear();
            } else {
                out.array_value.resize(array_len);
                for (uint64_t i = 0; i < array_len; ++i) {
                    if (!read_metadata_value(file, static_cast<GGUFMetadataType>(array_type), out.array_value[i])) {
                        return false;
                    }
                }
            }
            break;
        }
        default:
            Logger::instance().error("Unknown metadata type: " + std::to_string(static_cast<uint32_t>(type)));
            return false;
    }
    
    return true;
}

bool GGUFParser::parse(const std::string& file_path) {
    file_path_ = file_path;
    valid_ = false;
    
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        Logger::instance().error("Failed to open GGUF file: " + file_path);
        return false;
    }

    // Use a large read buffer so metadata parsing doesn't hammer the OS
    // with thousands of tiny syscalls (default buffer is only ~512 bytes on Windows)
    static thread_local std::vector<char> read_buf(2 * 1024 * 1024); // 2 MB
    file.rdbuf()->pubsetbuf(read_buf.data(), read_buf.size());
    
    // Read header
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != GGUF_MAGIC) {
        Logger::instance().error("Invalid GGUF magic number");
        return false;
    }
    
    file.read(reinterpret_cast<char*>(&version_), 4);
    Logger::instance().info("GGUF version: " + std::to_string(version_));
    
    file.read(reinterpret_cast<char*>(&tensor_count_), 8);
    file.read(reinterpret_cast<char*>(&metadata_count_), 8);
    
    Logger::instance().info("Tensor count: " + std::to_string(tensor_count_));
    Logger::instance().info("Metadata count: " + std::to_string(metadata_count_));
    
    // Read metadata
    for (uint64_t i = 0; i < metadata_count_; ++i) {
        std::string key = read_string(file);
        
        uint32_t value_type;
        file.read(reinterpret_cast<char*>(&value_type), 4);
        
        GGUFMetadataValue value;
        if (!read_metadata_value(file, static_cast<GGUFMetadataType>(value_type), value)) {
            Logger::instance().error("Failed to read metadata value for key: " + key);
            return false;
        }
        
        metadata_[key] = value;
    }
    
    Logger::instance().debug("Parsed " + std::to_string(metadata_.size()) + " metadata entries");
    
    // Read tensor infos
    tensor_infos_.reserve(tensor_count_);
    for (uint64_t i = 0; i < tensor_count_; ++i) {
        GGUFTensorInfo info;
        info.name = read_string(file);
        
        // Read number of dimensions
        uint32_t n_dims;
        file.read(reinterpret_cast<char*>(&n_dims), 4);
        
        // Read dimensions (reversed in file)
        info.dimensions.resize(n_dims);
        for (uint32_t j = 0; j < n_dims; ++j) {
            uint64_t dim;
            file.read(reinterpret_cast<char*>(&dim), 8);
            info.dimensions[n_dims - 1 - j] = dim; // Reverse
        }
        
        // Read tensor type
        uint32_t tensor_type;
        file.read(reinterpret_cast<char*>(&tensor_type), 4);
        info.type = static_cast<GGUFTensorType>(tensor_type);
        
        // Read offset (in bytes from tensor_data_offset)
        file.read(reinterpret_cast<char*>(&info.offset), 8);
        
        tensor_infos_.push_back(info);
    }
    
    // Calculate alignment (tensors are aligned to 32 bytes)
    uint64_t current_pos = file.tellg();
    tensor_data_offset_ = (current_pos + 31) & ~31ULL; // Align to 32 bytes
    
    Logger::instance().debug("Tensor data starts at offset: " + std::to_string(tensor_data_offset_));
    Logger::instance().info("✅ GGUF parsed successfully: " + std::to_string(tensor_count_) + " tensors");
    
    // Debug: Print first 10 tensor names
    Logger::instance().info("Sample tensor names:");
    int name_count = 0;
    for (const auto& info : tensor_infos_) {
        if (name_count++ < 10) {
            Logger::instance().info("  - " + info.name);
        }
    }
    
    valid_ = true;
    return true;
}

bool GGUFParser::get_metadata(const std::string& key, GGUFMetadataValue& out) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) {
        return false;
    }
    out = it->second;
    return true;
}

std::string GGUFParser::get_string(const std::string& key, const std::string& default_val) const {
    GGUFMetadataValue val;
    if (get_metadata(key, val) && val.type == GGUFMetadataType::STRING) {
        return val.string_value;
    }
    return default_val;
}

uint64_t GGUFParser::get_uint(const std::string& key, uint64_t default_val) const {
    GGUFMetadataValue val;
    if (get_metadata(key, val)) {
        switch (val.type) {
            case GGUFMetadataType::UINT8:
            case GGUFMetadataType::UINT16:
            case GGUFMetadataType::UINT32:
            case GGUFMetadataType::UINT64:
                return val.uint_value;
            case GGUFMetadataType::INT8:
            case GGUFMetadataType::INT16:
            case GGUFMetadataType::INT32:
            case GGUFMetadataType::INT64:
                return static_cast<uint64_t>(val.int_value);
            default:
                return default_val;
        }
    }
    return default_val;
}

int64_t GGUFParser::get_int(const std::string& key, int64_t default_val) const {
    GGUFMetadataValue val;
    if (get_metadata(key, val)) {
        switch (val.type) {
            case GGUFMetadataType::INT8:
            case GGUFMetadataType::INT16:
            case GGUFMetadataType::INT32:
            case GGUFMetadataType::INT64:
                return val.int_value;
            case GGUFMetadataType::UINT8:
            case GGUFMetadataType::UINT16:
            case GGUFMetadataType::UINT32:
            case GGUFMetadataType::UINT64:
                return static_cast<int64_t>(val.uint_value);
            default:
                return default_val;
        }
    }
    return default_val;
}

float GGUFParser::get_float(const std::string& key, float default_val) const {
    GGUFMetadataValue val;
    if (get_metadata(key, val)) {
        switch (val.type) {
            case GGUFMetadataType::FLOAT32:
            case GGUFMetadataType::FLOAT64:
                return static_cast<float>(val.float_value);
            case GGUFMetadataType::UINT8:
            case GGUFMetadataType::UINT16:
            case GGUFMetadataType::UINT32:
            case GGUFMetadataType::UINT64:
                return static_cast<float>(val.uint_value);
            default:
                return default_val;
        }
    }
    return default_val;
}

const GGUFTensorInfo* GGUFParser::find_tensor(const std::string& name) const {
    for (const auto& info : tensor_infos_) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

size_t GGUFParser::tensor_type_size(GGUFTensorType type) const {
    // Simplified - for quantized types this is approximate
    switch (type) {
        case GGUFTensorType::F32: return 4;
        case GGUFTensorType::F16: return 2;
        case GGUFTensorType::I32: return 4;
        case GGUFTensorType::I16: return 2;
        case GGUFTensorType::I8: return 1;
        default: return 4; // Approximation for quantized
    }
}

Tensor GGUFParser::load_tensor(const std::string& name) {
    if (!valid_) {
        throw std::runtime_error("GGUF parser not initialized");
    }
    
    const GGUFTensorInfo* info = find_tensor(name);
    if (!info) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    // Open file
    std::ifstream file(file_path_, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open GGUF file");
    }
    
    // Convert dimensions to TensorShape
    std::vector<int64_t> dims;
    for (auto d : info->dimensions) {
        dims.push_back(static_cast<int64_t>(d));
    }
    TensorShape shape(dims);
    
    // Convert type
    DType dtype = gguf_type_to_dtype(info->type);
    
    // Create tensor
    Tensor tensor = Tensor::empty(shape, dtype);
    
    // Seek to tensor data
    uint64_t file_offset = tensor_data_offset_ + info->offset;
    file.seekg(file_offset);
    
    // Read tensor data
    size_t bytes_to_read = tensor.size_bytes();
    file.read(reinterpret_cast<char*>(tensor.data()), bytes_to_read);
    
    if (!file) {
        throw std::runtime_error("Failed to read tensor data for: " + name);
    }
    
    Logger::instance().debug("Loaded tensor: " + name + " " + shape.to_string());
    
    return tensor;
}

std::unordered_map<std::string, Tensor> GGUFParser::load_all_tensors() {
    std::unordered_map<std::string, Tensor> tensors;
    
    Logger::instance().info("Loading all tensors from GGUF...");
    
    for (const auto& info : tensor_infos_) {
        try {
            tensors[info.name] = load_tensor(info.name);
        } catch (const std::exception& e) {
            Logger::instance().error("Failed to load tensor " + info.name + ": " + e.what());
        }
    }
    
    Logger::instance().info("✅ Loaded " + std::to_string(tensors.size()) + " tensors");
    
    return tensors;
}

// Architecture-specific getters
uint64_t GGUFParser::get_embedding_dim() const {
    std::string arch = get_architecture();
    return get_uint(arch + ".embedding_length", 3072);
}

uint64_t GGUFParser::get_num_layers() const {
    std::string arch = get_architecture();
    return get_uint(arch + ".block_count", 42);
}

uint64_t GGUFParser::get_num_heads() const {
    std::string arch = get_architecture();
    return get_uint(arch + ".attention.head_count", 16);
}

uint64_t GGUFParser::get_num_kv_heads() const {
    std::string arch = get_architecture();
    return get_uint(arch + ".attention.head_count_kv", 8);
}

uint64_t GGUFParser::get_vocab_size() const {
    uint64_t vocab_from_metadata = get_uint("tokenizer.ggml.token_count", 0);
    
    // If not in metadata, infer from token_embd.weight tensor shape
    if (vocab_from_metadata == 0) {
        auto tensor = find_tensor("token_embd.weight");
        if (tensor) {
            // First dimension is vocab size
            vocab_from_metadata = tensor->dimensions.at(0);
            Logger::instance().info("Inferred vocab_size from token_embd.weight: " + std::to_string(vocab_from_metadata));
        } else {
            vocab_from_metadata = 256000; // Ultimate fallback
        }
    }
    
    return vocab_from_metadata;
}

} // namespace ash
