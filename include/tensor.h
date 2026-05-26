#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace ash {

// Data types for tensors
enum class DType {
    F32,    // 32-bit float
    F16,    // 16-bit float
    Q8_0,   // 8-bit quantized (legacy GGUF, 32-elem blocks)
    Q4_0,   // 4-bit quantized (legacy GGUF, 32-elem blocks)
    Q4_K,   // 4-bit quantized (K-quants, 256-elem blocks)
    Q5_K,   // 5-bit quantized (K-quants, 256-elem blocks)
    Q6_K,   // 6-bit quantized (K-quants, 256-elem blocks)
    I32,    // 32-bit int
    I16,    // 16-bit int
    I8      // 8-bit int
};

// Tensor shape
struct TensorShape {
    std::vector<int64_t> dims;
    
    TensorShape() = default;
    TensorShape(std::initializer_list<int64_t> d) : dims(d) {}
    TensorShape(const std::vector<int64_t>& d) : dims(d) {}
    TensorShape(std::vector<int64_t>&& d) : dims(std::move(d)) {}
    
    // Get number of dimensions
    size_t ndim() const { return dims.size(); }
    
    // Get total number of elements
    int64_t numel() const;
    
    // Get size of specific dimension
    int64_t size(size_t dim) const { return dims[dim]; }
    
    // String representation
    std::string to_string() const;
};

// A tensor holds multi-dimensional data
class Tensor {
public:
    Tensor() = default;
    Tensor(TensorShape shape, DType dtype);
    ~Tensor();
    
    // No copy (tensors can be large)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // Move is OK
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Create tensor from data
    static Tensor from_data(const void* data, TensorShape shape, DType dtype);
    
    // Create empty tensor
    static Tensor empty(TensorShape shape, DType dtype);
    
    // Create zero-filled tensor
    static Tensor zeros(TensorShape shape, DType dtype);
    
    // Create one-filled tensor
    static Tensor ones(TensorShape shape, DType dtype);
    
    // Access
    const TensorShape& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size_bytes() const;
    
    // Get as specific type (for F32 tensors)
    float* data_f32() { return reinterpret_cast<float*>(data_); }
    const float* data_f32() const { return reinterpret_cast<const float*>(data_); }
    
    // Check if allocated
    bool is_allocated() const { return data_ != nullptr; }
    
    // Number of elements
    int64_t num_elements() const { return shape_.numel(); }
    
    // Clone tensor (deep copy)
    Tensor clone() const;
    
    // Memory locking (prevents OS from swapping to disk)
    bool lock_memory();
    void unlock_memory();
    bool is_memory_locked() const { return memory_locked_; }
    
    // Dequantize to F32 (if quantized)
    Tensor dequantize() const;
    
    // Print info
    std::string info() const;

private:
    TensorShape shape_;
    DType dtype_ = DType::F32;
    void* data_ = nullptr;
    bool owns_data_ = true;
    bool memory_locked_ = false;
    void* memory_lock_handle_ = nullptr;  // Opaque handle instead of unique_ptr
    
    // Allocate memory
    void allocate();
    void free();
};

// Get size of dtype in bytes (per element for non-quantized)
size_t dtype_size(DType dtype);

// Get name of dtype
const char* dtype_name(DType dtype);

// Dequantization functions (quantized → F32)
void dequantize_q8_0(const void* src, float* dst, int64_t n);
void dequantize_q4_0(const void* src, float* dst, int64_t n);
void dequantize_q4_k(const void* src, float* dst, int64_t n);
void dequantize_q5_k(const void* src, float* dst, int64_t n);
void dequantize_q6_k(const void* src, float* dst, int64_t n);

} // namespace ash
