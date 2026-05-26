#pragma once

#include "tensor.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <fstream>

namespace ash {

// GGUF metadata value types
enum class GGUFMetadataType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

// GGUF tensor type
enum class GGUFTensorType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18
};

// Convert GGUF type to our DType
DType gguf_type_to_dtype(GGUFTensorType gguf_type);

// Metadata value (variant-like storage)
struct GGUFMetadataValue {
    GGUFMetadataType type;
    std::string string_value;
    uint64_t uint_value;
    int64_t int_value;
    double float_value;
    bool bool_value;
    std::vector<GGUFMetadataValue> array_value;
};

// GGUF tensor info (before loading data)
struct GGUFTensorInfo {
    std::string name;
    GGUFTensorType type;
    std::vector<uint64_t> dimensions;
    uint64_t offset; // Offset in file where tensor data starts
    
    int64_t num_elements() const {
        int64_t n = 1;
        for (auto d : dimensions) n *= d;
        return n;
    }
};

// GGUF file parser
class GGUFParser {
public:
    GGUFParser();
    ~GGUFParser();
    
    // Parse GGUF file header and metadata
    bool parse(const std::string& file_path);
    
    // Get metadata value by key
    bool get_metadata(const std::string& key, GGUFMetadataValue& out) const;
    
    // Get metadata as specific type (convenience)
    std::string get_string(const std::string& key, const std::string& default_val = "") const;
    uint64_t get_uint(const std::string& key, uint64_t default_val = 0) const;
    int64_t get_int(const std::string& key, int64_t default_val = 0) const;
    float get_float(const std::string& key, float default_val = 0.0f) const;
    
    // Get all tensor infos
    const std::vector<GGUFTensorInfo>& get_tensors() const { return tensor_infos_; }
    
    // Find tensor by name
    const GGUFTensorInfo* find_tensor(const std::string& name) const;
    
    // Load specific tensor data
    Tensor load_tensor(const std::string& name);
    
    // Load all tensors into map
    std::unordered_map<std::string, Tensor> load_all_tensors();
    
    // Get model architecture info
    std::string get_architecture() const { return get_string("general.architecture"); }
    uint64_t get_context_length() const;
    uint64_t get_embedding_dim() const;
    uint64_t get_num_layers() const;
    uint64_t get_num_heads() const;
    uint64_t get_num_kv_heads() const;
    uint64_t get_vocab_size() const;
    
    // Check if file is valid GGUF
    bool is_valid() const { return valid_; }
    
    // Get file path
    const std::string& file_path() const { return file_path_; }

private:
    std::string file_path_;
    bool valid_ = false;
    
    uint32_t version_;
    uint64_t tensor_count_;
    uint64_t metadata_count_;
    
    std::unordered_map<std::string, GGUFMetadataValue> metadata_;
    std::vector<GGUFTensorInfo> tensor_infos_;
    uint64_t tensor_data_offset_; // Where tensor data starts in file
    
    // Helper: read metadata value from stream
    bool read_metadata_value(std::ifstream& file, GGUFMetadataType type, GGUFMetadataValue& out);
    
    // Helper: read string from stream
    std::string read_string(std::ifstream& file);
    
    // Helper: calculate size of tensor type
    size_t tensor_type_size(GGUFTensorType type) const;
};

} // namespace ash
