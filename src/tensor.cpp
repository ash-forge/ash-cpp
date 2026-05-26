#include "tensor.h"
#include "memory_lock.h"  // Full definition needed for ScopedMemoryLock
#include "logger.h"
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <cmath>

#ifdef _WIN32
#include <malloc.h> // For _aligned_malloc
#endif

namespace ash {

// TensorShape implementation
int64_t TensorShape::numel() const {
    if (dims.empty()) return 0;
    int64_t n = 1;
    for (auto d : dims) {
        n *= d;
    }
    return n;
}

std::string TensorShape::to_string() const {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << dims[i];
    }
    ss << "]";
    return ss.str();
}

// Tensor implementation
Tensor::Tensor(TensorShape shape, DType dtype)
    : shape_(shape), dtype_(dtype), data_(nullptr), owns_data_(true) {
    allocate();
}

Tensor::~Tensor() {
    free();
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_))
    , dtype_(other.dtype_)
    , data_(other.data_)
    , owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.owns_data_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free();
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

Tensor Tensor::from_data(const void* data, TensorShape shape, DType dtype) {
    Tensor t(shape, dtype);
    std::memcpy(t.data_, data, t.size_bytes());
    return t;
}

Tensor Tensor::empty(TensorShape shape, DType dtype) {
    return Tensor(shape, dtype);
}

Tensor Tensor::zeros(TensorShape shape, DType dtype) {
    Tensor t(shape, dtype);
    std::memset(t.data_, 0, t.size_bytes());
    return t;
}

Tensor Tensor::ones(TensorShape shape, DType dtype) {
    Tensor t(shape, dtype);
    if (dtype == DType::F32) {
        float* data = t.data_f32();
        int64_t n = t.shape().numel();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = 1.0f;
        }
    }
    return t;
}

size_t Tensor::size_bytes() const {
    int64_t n = shape_.numel();
    
    // For quantized types, size depends on block structure
    switch (dtype_) {
        case DType::F32: return n * 4;
        case DType::F16: return n * 2;
        case DType::I32: return n * 4;
        case DType::I16: return n * 2;
        case DType::I8:  return n;
        
        // Quantized types — exact block-aligned byte counts
        case DType::Q8_0: return (n / 32) * 34;   // 32 elems: 2B F16 scale + 32B int8
        case DType::Q4_0: return (n / 32) * 18;   // 32 elems: 2B F16 scale + 16B nibbles
        case DType::Q4_K: return (n / 256) * 144; // 256 elems: 2B d + 2B dmin + 12B scales + 128B nibbles
        case DType::Q5_K: return (n / 256) * 176; // 256 elems: 2B d + 2B dmin + 12B scales + 32B qh + 128B ql
        case DType::Q6_K: return (n / 256) * 210; // 256 elems: 128B ql + 64B qh + 16B scales + 2B d
        
        default:
            throw std::runtime_error("Unknown dtype");
    }
}

void Tensor::allocate() {
    if (data_) return;
    
    size_t bytes = size_bytes();
    if (bytes == 0) return;
    
    // Align to 64 bytes for SIMD
    #ifdef _WIN32
        data_ = _aligned_malloc(bytes, 64);
    #else
        data_ = std::aligned_alloc(64, bytes);
    #endif
    
    if (!data_) {
        throw std::runtime_error("Failed to allocate tensor memory");
    }
    
    owns_data_ = true;
}

void Tensor::free() {
    if (data_ && owns_data_) {
        #ifdef _WIN32
            _aligned_free(data_);
        #else
            std::free(data_);
        #endif
        data_ = nullptr;
    }
}

Tensor Tensor::dequantize() const {
    if (dtype_ == DType::F32) {
        // Already F32, just copy
        return Tensor::from_data(data_, shape_, DType::F32);
    }
    
    // Create F32 output tensor
    Tensor result = Tensor::empty(shape_, DType::F32);
    int64_t n = shape_.numel();
    
    // Dequantize based on type
    switch (dtype_) {
        case DType::Q8_0:
            dequantize_q8_0(data_, result.data_f32(), n);
            break;
        case DType::Q4_0:
            dequantize_q4_0(data_, result.data_f32(), n);
            break;
        case DType::Q4_K:
            dequantize_q4_k(data_, result.data_f32(), n);
            break;
        case DType::Q5_K:
            dequantize_q5_k(data_, result.data_f32(), n);
            break;
        case DType::Q6_K:
            dequantize_q6_k(data_, result.data_f32(), n);
            break;
        case DType::F16:
            // TODO: F16 → F32 conversion
            throw std::runtime_error("F16 dequantization not yet implemented");
        default:
            throw std::runtime_error("Cannot dequantize this dtype");
    }
    
    return result;
}

std::string Tensor::info() const {
    std::stringstream ss;
    ss << "Tensor(shape=" << shape_.to_string();
    ss << ", dtype=" << dtype_name(dtype_);
    ss << ", bytes=" << size_bytes();
    ss << ", allocated=" << (data_ != nullptr ? "yes" : "no");
    ss << ")";
    return ss.str();
}

Tensor Tensor::clone() const {
    if (!is_allocated()) {
        return Tensor();
    }
    
    Tensor copy = Tensor::empty(shape_, dtype_);
    std::memcpy(copy.data(), data_, size_bytes());
    return copy;
}

bool Tensor::lock_memory() {
    if (!data_ || !owns_data_) {
        return false;
    }
    
    if (memory_locked_) {
        return true;  // Already locked
    }
    
    size_t bytes = size_bytes();
    memory_lock_handle_ = new ScopedMemoryLock(data_, bytes);
    memory_locked_ = static_cast<ScopedMemoryLock*>(memory_lock_handle_)->is_locked();
    
    if (!memory_locked_) {
        Logger::instance().warning("Failed to lock tensor memory: " + 
            static_cast<ScopedMemoryLock*>(memory_lock_handle_)->error());
        delete static_cast<ScopedMemoryLock*>(memory_lock_handle_);
        memory_lock_handle_ = nullptr;
    } else {
        Logger::instance().debug("Locked tensor memory: " + std::to_string(bytes) + " bytes");
    }
    
    return memory_locked_;
}

void Tensor::unlock_memory() {
    if (memory_lock_handle_) {
        delete static_cast<ScopedMemoryLock*>(memory_lock_handle_);
        memory_lock_handle_ = nullptr;
        memory_locked_ = false;
    }
}

// Dtype utilities
size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::I32: return 4;
        case DType::I16: return 2;
        case DType::I8:  return 1;
        // Quantized types don't have a fixed per-element size
        default: return 0;
    }
}

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::F32: return "float32";
        case DType::F16: return "float16";
        case DType::Q8_0: return "Q8_0";
        case DType::Q4_0: return "Q4_0";
        case DType::Q4_K: return "Q4_K";
        case DType::Q5_K: return "Q5_K";
        case DType::Q6_K: return "Q6_K";
        case DType::I32: return "int32";
        case DType::I16: return "int16";
        case DType::I8:  return "int8";
        default: return "unknown";
    }
}

// Dequantization implementations
// Reference: llama.cpp ggml-quants.c (MIT license)

// Shared F16→F32 conversion helper (IEEE 754 compliant)
static inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h >> 15) & 0x1;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            uint32_t e = 127 - 14, m = mant;
            while ((m & 0x400) == 0) { m <<= 1; --e; }
            m &= 0x3FF;
            f = (sign << 31) | (e << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float result; std::memcpy(&result, &f, 4); return result;
}

// Extract 6-bit scale/min pair for sub-block j (0–7) from 12-byte K-quant scales array.
// Used by Q4_K and Q5_K.
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* sc, uint8_t* m) {
    if (j < 4) {
        *sc = q[j]     & 0x3F;
        *m  = q[j + 4] & 0x3F;
    } else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

void dequantize_q8_0(const void* src, float* dst, int64_t n) {
    // Block: 2B F16 scale + 32 × int8 = 34 bytes per 32 elements
    const int BS = 32;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
    const int64_t nb = n / BS;
    for (int64_t b = 0; b < nb; ++b) {
        const float d = f16_to_f32(*reinterpret_cast<const uint16_t*>(p));
        p += 2;
        const int8_t* qs = reinterpret_cast<const int8_t*>(p);
        for (int j = 0; j < BS; ++j)
            dst[b * BS + j] = d * static_cast<float>(qs[j]);
        p += BS;
    }
}

void dequantize_q4_0(const void* src, float* dst, int64_t n) {
    // Block: 2B F16 scale + 16B nibbles = 18 bytes per 32 elements
    // Each nibble x is in [0,15]; dequant: (x - 8) * scale
    const int BS = 32;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
    const int64_t nb = n / BS;
    for (int64_t b = 0; b < nb; ++b) {
        const float d = f16_to_f32(*reinterpret_cast<const uint16_t*>(p));
        p += 2;
        for (int j = 0; j < BS / 2; ++j) {
            const uint8_t byte = p[j];
            dst[b * BS + j * 2]     = d * (static_cast<int>( byte       & 0x0F) - 8);
            dst[b * BS + j * 2 + 1] = d * (static_cast<int>((byte >> 4) & 0x0F) - 8);
        }
        p += BS / 2;
    }
}

void dequantize_q4_k(const void* src, float* dst, int64_t n) {
    // Block: 2B d + 2B dmin + 12B scales + 128B nibbles = 144 bytes per 256 elements
    // 8 sub-blocks of 32 elements; each has 6-bit scale and 6-bit min
    const int BS = 256;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
    const int64_t nb = n / BS;
    for (int64_t b = 0; b < nb; ++b) {
        const float d    = f16_to_f32(*reinterpret_cast<const uint16_t*>(p));
        const float dmin = f16_to_f32(*reinterpret_cast<const uint16_t*>(p + 2));
        const uint8_t* scales = p + 4;
        const uint8_t* qs     = p + 4 + 12;
        p += 144;

        float* y = dst + b * BS;
        int is = 0;
        const uint8_t* q = qs;
        for (int j = 0; j < BS; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) y[l]      = d1 * (q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l) y[32 + l] = d2 * (q[l] >> 4)   - m2;
            y += 64; q += 32; is += 2;
        }
    }
}

void dequantize_q5_k(const void* src, float* dst, int64_t n) {
    // Block: 2B d + 2B dmin + 12B scales + 32B qh + 128B ql = 176 bytes per 256 elements
    // 5-bit: lower 4 bits in ql nibbles, upper bit in qh
    const int BS = 256;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
    const int64_t nb = n / BS;
    for (int64_t b = 0; b < nb; ++b) {
        const float d    = f16_to_f32(*reinterpret_cast<const uint16_t*>(p));
        const float dmin = f16_to_f32(*reinterpret_cast<const uint16_t*>(p + 2));
        const uint8_t* scales = p + 4;
        const uint8_t* qh     = p + 4 + 12;       // 32 bytes of high bits
        const uint8_t* ql     = p + 4 + 12 + 32;  // 128 bytes of low nibbles
        p += 176;

        float* y = dst + b * BS;
        int is = 0;
        const uint8_t* q  = ql;
        const uint8_t* hb = qh;
        uint32_t u1 = 1, u2 = 2;
        for (int j = 0; j < BS; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) {
                y[l]      = d1 * ((q[l] & 0x0F) + (hb[l/8] & u1 ? 16 : 0)) - m1;
                u1 <<= 2;
            }
            for (int l = 0; l < 32; ++l) {
                y[32 + l] = d2 * ((q[l] >> 4)   + (hb[l/8] & u2 ? 16 : 0)) - m2;
                u2 <<= 2;
            }
            y += 64; q += 32; is += 2;
            hb += 4; u1 = 1; u2 = 2;
        }
    }
}

void dequantize_q6_k(const void* src, float* dst, int64_t n) {
    // Block: 128B ql (nibbles, lower 4 bits) + 64B qh (2 upper bits) + 16B int8 scales + 2B F16 d
    // = 210 bytes per 256 elements
    const int BS = 256;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
    const int64_t nb = n / BS;
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t* ql     = p;           // 128 bytes
        const uint8_t* qh     = p + 128;     // 64 bytes
        const int8_t*  sc     = reinterpret_cast<const int8_t*>(p + 128 + 64);  // 16 bytes
        const float    d      = f16_to_f32(*reinterpret_cast<const uint16_t*>(p + 128 + 64 + 16));
        p += 210;

        float* y = dst + b * BS;
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            // 6-bit values: 4 lower bits from ql, 2 upper bits from qh (2 per byte)
            y[l +   0] = d * sc[is + 0] * ((static_cast<int>( ql[l]        & 0x0F) | ((qh[l/4]       & 0x03) << 4)) - 32);
            y[l +  32] = d * sc[is + 2] * ((static_cast<int>( ql[l + 32]   & 0x0F) | ((qh[l/4 + 8]   & 0x03) << 4)) - 32);
            y[l +  64] = d * sc[is + 4] * ((static_cast<int>((ql[l]        >> 4)   | ((qh[l/4]       & 0x0C) << 2)) - 32));
            y[l +  96] = d * sc[is + 6] * ((static_cast<int>((ql[l + 32]   >> 4)   | ((qh[l/4 + 8]   & 0x0C) << 2)) - 32));
            y[l + 128] = d * sc[is + 8] * ((static_cast<int>( ql[l + 64]   & 0x0F) | ((qh[l/4 + 16]  & 0x03) << 4)) - 32);
            y[l + 160] = d * sc[is +10] * ((static_cast<int>( ql[l + 96]   & 0x0F) | ((qh[l/4 + 24]  & 0x03) << 4)) - 32);
            y[l + 192] = d * sc[is +12] * ((static_cast<int>((ql[l + 64]   >> 4)   | ((qh[l/4 + 16]  & 0x0C) << 2)) - 32));
            y[l + 224] = d * sc[is +14] * ((static_cast<int>((ql[l + 96]   >> 4)   | ((qh[l/4 + 24]  & 0x0C) << 2)) - 32));
        }
    }
}

} // namespace ash
