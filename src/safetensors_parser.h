#pragma once

#include "tensor.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace ash {

/**
 * @brief Metadata for a tensor in a safetensors file
 */
struct SafeTensorInfo {
    std::string name;
    std::string dtype;  // "F32", "F16", "BF16", "I32", "I64", etc.
    std::vector<uint64_t> shape;
    uint64_t data_offset_start;  // Start offset in data section
    uint64_t data_offset_end;    // End offset in data section
};

/**
 * @brief Parser for SafeTensors file format
 * 
 * SafeTensors format:
 * - 8 bytes: header length (uint64, little-endian)
 * - N bytes: JSON header (UTF-8)
 * - Remaining: tensor data
 */
class SafeTensorsParser {
public:
    SafeTensorsParser();
    ~SafeTensorsParser();
    
    /**
     * @brief Parse a safetensors file
     * @param file_path Path to .safetensors file
     * @return true if successful
     */
    bool parse(const std::string& file_path);
    
    /**
     * @brief Get list of all tensor names
     */
    std::vector<std::string> list_tensors() const;
    
    /**
     * @brief Find tensor metadata by name
     * @return Pointer to tensor info, or nullptr if not found
     */
    const SafeTensorInfo* find_tensor(const std::string& name) const;
    
    /**
     * @brief Load a tensor by name
     * @param name Tensor name
     * @return Tensor object with data
     */
    Tensor load_tensor(const std::string& name);
    
    /**
     * @brief Load all tensors from the file (one file open per tensor)
     * @return Map of tensor name to Tensor object
     */
    std::unordered_map<std::string, Tensor> load_all_tensors();

    /**
     * @brief Load all tensors in a single sequential pass (faster for bulk loads)
     * @return Map of tensor name to Tensor object
     */
    std::unordered_map<std::string, Tensor> load_all_tensors_fast();
    
    /**
     * @brief Check if parser is valid (file was parsed successfully)
     */
    bool is_valid() const { return valid_; }
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    std::string file_path_;
    std::vector<SafeTensorInfo> tensor_infos_;
    uint64_t header_size_;  // Size of JSON header
    uint64_t data_offset_;  // Offset where tensor data starts (8 + header_size_)
    bool valid_;
    
    /**
     * @brief Convert safetensors dtype string to DType enum
     */
    DType dtype_from_string(const std::string& dtype_str) const;
};

} // namespace ash
