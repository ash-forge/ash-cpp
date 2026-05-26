#include "safetensors_parser.h"
#include "logger.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ash {

struct SafeTensorsParser::Impl {
    // Empty for now, can add caching or other state later
};

SafeTensorsParser::SafeTensorsParser() 
    : impl_(std::make_unique<Impl>()), 
      header_size_(0),
      data_offset_(0),
      valid_(false) {
}

SafeTensorsParser::~SafeTensorsParser() = default;

bool SafeTensorsParser::parse(const std::string& file_path) {
    file_path_ = file_path;
    tensor_infos_.clear();
    valid_ = false;
    
    Logger::instance().info("Parsing SafeTensors file: " + file_path);
    
    // Open file
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        Logger::instance().error("Failed to open SafeTensors file: " + file_path);
        return false;
    }
    
    // Read header length (first 8 bytes, little-endian uint64)
    uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!file) {
        Logger::instance().error("Failed to read header length");
        return false;
    }
    
    header_size_ = header_len;
    data_offset_ = 8 + header_len;
    
    Logger::instance().debug("SafeTensors header size: " + std::to_string(header_len) + " bytes");
    
    // Read JSON header
    std::vector<char> header_data(header_len);
    file.read(header_data.data(), header_len);
    if (!file) {
        Logger::instance().error("Failed to read header data");
        return false;
    }
    
    // Parse JSON
    json header;
    try {
        header = json::parse(header_data.begin(), header_data.end());
    } catch (const json::parse_error& e) {
        Logger::instance().error("JSON parse error: " + std::string(e.what()));
        return false;
    }
    
    // Extract tensor metadata
    for (auto& [key, value] : header.items()) {
        // Skip metadata entry (usually "__metadata__")
        if (key.find("__") == 0) {
            continue;
        }
        
        SafeTensorInfo info;
        info.name = key;
        
        try {
            info.dtype = value["dtype"].get<std::string>();
            info.shape = value["shape"].get<std::vector<uint64_t>>();
            
            auto offsets = value["data_offsets"].get<std::vector<uint64_t>>();
            if (offsets.size() != 2) {
                Logger::instance().error("Invalid data_offsets for tensor: " + key);
                continue;
            }
            info.data_offset_start = offsets[0];
            info.data_offset_end = offsets[1];
            
            tensor_infos_.push_back(info);
        } catch (const json::exception& e) {
            Logger::instance().error("Error parsing tensor metadata for " + key + ": " + e.what());
            continue;
        }
    }
    
    Logger::instance().info("✅ Parsed " + std::to_string(tensor_infos_.size()) + " tensors from SafeTensors");
    valid_ = true;
    return true;
}

std::vector<std::string> SafeTensorsParser::list_tensors() const {
    std::vector<std::string> names;
    names.reserve(tensor_infos_.size());
    for (const auto& info : tensor_infos_) {
        names.push_back(info.name);
    }
    return names;
}

const SafeTensorInfo* SafeTensorsParser::find_tensor(const std::string& name) const {
    for (const auto& info : tensor_infos_) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

DType SafeTensorsParser::dtype_from_string(const std::string& dtype_str) const {
    if (dtype_str == "F32") return DType::F32;
    if (dtype_str == "F16") return DType::F16;
    if (dtype_str == "BF16") {
        // BF16 not natively supported, will convert to F32 on load
        Logger::instance().debug("BF16 tensor will be converted to F32 on load");
        return DType::F32;  // Treat as F32, caller must convert
    }
    if (dtype_str == "I32") return DType::I32;
    if (dtype_str == "I8") return DType::I8;
    
    Logger::instance().error("Unknown/unsupported dtype: " + dtype_str);
    return DType::F32;  // Default fallback
}

Tensor SafeTensorsParser::load_tensor(const std::string& name) {
    if (!valid_) {
        throw std::runtime_error("SafeTensorsParser not initialized");
    }
    
    const SafeTensorInfo* info = find_tensor(name);
    if (!info) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    // Open file
    std::ifstream file(file_path_, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open SafeTensors file");
    }
    
    // Convert shape to TensorShape
    std::vector<int64_t> dims;
    for (auto d : info->shape) {
        dims.push_back(static_cast<int64_t>(d));
    }
    TensorShape shape(dims);
    
    // Check if original dtype is BF16 (needs conversion)
    bool is_bf16 = (info->dtype == "BF16");
    
    if (is_bf16) {
        Logger::instance().info("Converting BF16 tensor: " + name);
    }
    
    // Get dtype (BF16 will be converted to F32)
    DType dtype = dtype_from_string(info->dtype);
    
    // Calculate data size
    size_t data_size = info->data_offset_end - info->data_offset_start;
    
    if (is_bf16) {
        // BF16: each element is 2 bytes, but we'll convert to F32 (4 bytes each)
        size_t num_elements = data_size / 2;  // BF16 is 2 bytes per element
        
        // Read BF16 data
        std::vector<uint16_t> bf16_data(num_elements);
        uint64_t file_offset = data_offset_ + info->data_offset_start;
        file.seekg(file_offset);
        file.read(reinterpret_cast<char*>(bf16_data.data()), data_size);
        
        if (!file) {
            throw std::runtime_error("Failed to read BF16 tensor data for: " + name);
        }
        
        // Create F32 tensor
        Tensor tensor = Tensor::empty(shape, DType::F32);
        float* f32_data = tensor.data_f32();
        
        // Convert BF16 to F32
        // BF16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
        // F32 format: 1 sign bit, 8 exponent bits, 23 mantissa bits
        // Conversion: BF16 is just the top 16 bits of F32, so left-shift by 16
        for (size_t i = 0; i < num_elements; i++) {
            uint32_t f32_bits = static_cast<uint32_t>(bf16_data[i]) << 16;
            std::memcpy(&f32_data[i], &f32_bits, sizeof(float));
        }
        
        Logger::instance().debug("Loaded & converted BF16->F32 tensor: " + name + " " + shape.to_string());
        return tensor;
    }
    
    // Non-BF16 path (F32, F16, etc.)
    Tensor tensor = Tensor::empty(shape, dtype);
    size_t expected_size = tensor.size_bytes();
    
    if (data_size != expected_size) {
        Logger::instance().error(
            "Size mismatch for " + name + ": file has " + 
            std::to_string(data_size) + " bytes, expected " + 
            std::to_string(expected_size)
        );
        throw std::runtime_error("Tensor size mismatch");
    }
    
    // Seek to tensor data
    uint64_t file_offset = data_offset_ + info->data_offset_start;
    file.seekg(file_offset);
    
    // Read tensor data
    file.read(reinterpret_cast<char*>(tensor.data()), data_size);
    if (!file) {
        throw std::runtime_error("Failed to read tensor data for: " + name);
    }
    
    Logger::instance().debug("Loaded tensor: " + name + " " + shape.to_string());
    
    return tensor;
}

std::unordered_map<std::string, Tensor> SafeTensorsParser::load_all_tensors() {
    std::unordered_map<std::string, Tensor> tensors;
    
    Logger::instance().info("Loading all tensors from SafeTensors...");
    
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

std::unordered_map<std::string, Tensor> SafeTensorsParser::load_all_tensors_fast() {
    std::unordered_map<std::string, Tensor> result;
    
    if (!valid_) {
        throw std::runtime_error("SafeTensorsParser not initialized");
    }
    
    Logger::instance().info("Bulk-loading " + std::to_string(tensor_infos_.size()) + " tensors from: " + file_path_);
    
    // Sort tensors by file offset for sequential I/O
    std::vector<const SafeTensorInfo*> ordered;
    ordered.reserve(tensor_infos_.size());
    for (const auto& info : tensor_infos_) {
        ordered.push_back(&info);
    }
    std::sort(ordered.begin(), ordered.end(),
        [](const SafeTensorInfo* a, const SafeTensorInfo* b) {
            return a->data_offset_start < b->data_offset_start;
        });
    
    std::ifstream file(file_path_, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + file_path_);
    }
    
    for (const SafeTensorInfo* info : ordered) {
        std::vector<int64_t> dims;
        for (auto d : info->shape) dims.push_back(static_cast<int64_t>(d));
        TensorShape shape(dims);
        
        bool is_bf16 = (info->dtype == "BF16");
        size_t data_size = info->data_offset_end - info->data_offset_start;
        uint64_t file_offset = data_offset_ + info->data_offset_start;
        
        file.seekg(file_offset);
        
        if (is_bf16) {
            size_t num_elements = data_size / 2;
            std::vector<uint16_t> bf16_data(num_elements);
            file.read(reinterpret_cast<char*>(bf16_data.data()), data_size);
            if (!file) throw std::runtime_error("Failed to read: " + info->name);
            
            Tensor tensor = Tensor::empty(shape, DType::F32);
            float* f32_data = tensor.data_f32();
            for (size_t i = 0; i < num_elements; i++) {
                uint32_t bits = static_cast<uint32_t>(bf16_data[i]) << 16;
                std::memcpy(&f32_data[i], &bits, sizeof(float));
            }
            result[info->name] = std::move(tensor);
        } else {
            DType dtype = dtype_from_string(info->dtype);
            Tensor tensor = Tensor::empty(shape, dtype);
            file.read(reinterpret_cast<char*>(tensor.data()), data_size);
            if (!file) throw std::runtime_error("Failed to read: " + info->name);
            result[info->name] = std::move(tensor);
        }
    }
    
    Logger::instance().info("✅ Bulk-loaded " + std::to_string(result.size()) + " tensors");
    return result;
}

} // namespace ash
