#include "matrix_ops.h"
#include "logger.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace ash {

// Helper: check tensor is 2D
static void check_2d(const Tensor& t, const char* name) {
    if (t.shape().ndim() != 2) {
        throw std::runtime_error(std::string(name) + " must be 2D, got " + 
                                 std::to_string(t.shape().ndim()) + "D");
    }
}

// Matrix multiplication: C = A @ B
Tensor matmul(const Tensor& a, const Tensor& b) {
    check_2d(a, "A");
    check_2d(b, "B");
    
    int64_t m = a.shape().size(0);
    int64_t k = a.shape().size(1);
    int64_t k2 = b.shape().size(0);
    int64_t n = b.shape().size(1);
    
    if (k != k2) {
        throw std::runtime_error("matmul dimension mismatch: A[" + 
                                 std::to_string(m) + "," + std::to_string(k) + "] @ B[" +
                                 std::to_string(k2) + "," + std::to_string(n) + "]");
    }
    
    // All weights are pre-dequantized at load, activations are created as F32
    // Verify tensors are F32 (safety check)
    if (a.dtype() != DType::F32 || b.dtype() != DType::F32) {
        throw std::runtime_error("matmul: non-F32 tensors not supported (A=" + 
                                std::to_string(static_cast<int>(a.dtype())) + ", B=" + 
                                std::to_string(static_cast<int>(b.dtype())) + ")");
    }
    
    const float* a_data = a.data_f32();
    const float* b_data = b.data_f32();
    
    // Allocate output
    Tensor c = Tensor::zeros({m, n}, DType::F32);
    float* c_data = c.data_f32();
    
    // Naive implementation, parallelized over rows×cols via OpenMP
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int64_t p = 0; p < k; ++p) {
                sum += a_data[i * k + p] * b_data[p * n + j];
            }
            c_data[i * n + j] = sum;
        }
    }
    
    return c;
}

// Matrix multiplication with transpose support
// This is MUCH faster than explicit transpose + matmul
// Computes: C = op(A) @ op(B) where op(X) is either X or X^T
Tensor matmul_transposed(const Tensor& a, const Tensor& b, bool transpose_a, bool transpose_b) {
    check_2d(a, "A");
    check_2d(b, "B");
    
    // Get dimensions based on transpose flags
    int64_t m = transpose_a ? a.shape().size(1) : a.shape().size(0);
    int64_t k_a = transpose_a ? a.shape().size(0) : a.shape().size(1);
    int64_t k_b = transpose_b ? b.shape().size(1) : b.shape().size(0);
    int64_t n = transpose_b ? b.shape().size(0) : b.shape().size(1);
    
    if (k_a != k_b) {
        throw std::runtime_error("matmul_transposed dimension mismatch: op(A) has k=" + 
                               std::to_string(k_a) + ", op(B) has k=" + std::to_string(k_b));
    }
    int64_t k = k_a;
    
    // Verify tensors are F32
    if (a.dtype() != DType::F32 || b.dtype() != DType::F32) {
        throw std::runtime_error("matmul_transposed: non-F32 tensors");
    }
    
    const float* a_data = a.data_f32();
    const float* b_data = b.data_f32();
    
    // Original dimensions of A and B for indexing
    int64_t a_rows = a.shape().size(0);
    int64_t a_cols = a.shape().size(1);
    int64_t b_rows = b.shape().size(0);
    int64_t b_cols = b.shape().size(1);
    
    // Allocate output: [m, n]
    Tensor c = Tensor::zeros({m, n}, DType::F32);
    float* c_data = c.data_f32();
    
    // Optimized loops based on transpose flags
    // This avoids creating temporary transposed matrices
    if (!transpose_a && transpose_b) {
        // Most common case: A @ B^T
        // A: [m, k], B: [n, k] -> C: [m, n]
        // collapse(2) parallelizes across both i and j — critical for m=1 (generation)
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    // A[i, p] * B[j, p]  (B transposed)
                    sum += a_data[i * a_cols + p] * b_data[j * b_cols + p];
                }
                c_data[i * n + j] = sum;
            }
        }
    } else if (transpose_a && !transpose_b) {
        // A^T @ B
        // A: [k, m], B: [k, n] -> C: [m, n]
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    // A[p, i] * B[p, j]  (A transposed)
                    sum += a_data[p * a_cols + i] * b_data[p * b_cols + j];
                }
                c_data[i * n + j] = sum;
            }
        }
    } else if (transpose_a && transpose_b) {
        // A^T @ B^T
        // A: [k, m], B: [n, k] -> C: [m, n]
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    // A[p, i] * B[j, p]  (both transposed)
                    sum += a_data[p * a_cols + i] * b_data[j * b_cols + p];
                }
                c_data[i * n + j] = sum;
            }
        }
    } else {
        // No transpose - use standard matmul
        return matmul(a, b);
    }
    
    return c;
}

// Matrix-vector multiplication
Tensor matvec(const Tensor& a, const Tensor& x) {
    check_2d(a, "A");
    if (x.shape().ndim() != 1) {
        throw std::runtime_error("x must be 1D vector");
    }
    
    int64_t m = a.shape().size(0);
    int64_t k = a.shape().size(1);
    
    if (k != x.shape().size(0)) {
        throw std::runtime_error("matvec dimension mismatch");
    }
    
    // Verify tensors are F32
    if (a.dtype() != DType::F32 || x.dtype() != DType::F32) {
        throw std::runtime_error("matvec: non-F32 tensors");
    }
    
    const float* a_data = a.data_f32();
    const float* x_data = x.data_f32();
    
    Tensor y = Tensor::zeros({m}, DType::F32);
    float* y_data = y.data_f32();
    
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; ++i) {
        float sum = 0.0f;
        for (int64_t j = 0; j < k; ++j) {
            sum += a_data[i * k + j] * x_data[j];
        }
        y_data[i] = sum;
    }
    
    return y;
}

// Element-wise add
Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape().numel() != b.shape().numel()) {
        throw std::runtime_error("add: shape mismatch");
    }
    
    int64_t n = a.shape().numel();
    Tensor c = Tensor::empty(a.shape(), DType::F32);
    
    const float* a_data = a.data_f32();
    const float* b_data = b.data_f32();
    float* c_data = c.data_f32();
    
    for (int64_t i = 0; i < n; ++i) {
        c_data[i] = a_data[i] + b_data[i];
    }
    
    return c;
}

// Element-wise multiply
Tensor multiply(const Tensor& a, const Tensor& b) {
    if (a.shape().numel() != b.shape().numel()) {
        throw std::runtime_error("multiply: shape mismatch");
    }
    
    int64_t n = a.shape().numel();
    Tensor c = Tensor::empty(a.shape(), DType::F32);
    
    const float* a_data = a.data_f32();
    const float* b_data = b.data_f32();
    float* c_data = c.data_f32();
    
    for (int64_t i = 0; i < n; ++i) {
        c_data[i] = a_data[i] * b_data[i];
    }
    
    return c;
}

// Scale by scalar
Tensor scale(const Tensor& a, float scalar) {
    int64_t n = a.shape().numel();
    Tensor c = Tensor::empty(a.shape(), DType::F32);
    
    const float* a_data = a.data_f32();
    float* c_data = c.data_f32();
    
    for (int64_t i = 0; i < n; ++i) {
        c_data[i] = a_data[i] * scalar;
    }
    
    return c;
}

// ReLU activation
Tensor relu(const Tensor& x) {
    int64_t n = x.shape().numel();
    Tensor y = Tensor::empty(x.shape(), DType::F32);
    
    const float* x_data = x.data_f32();
    float* y_data = y.data_f32();
    
    for (int64_t i = 0; i < n; ++i) {
        y_data[i] = std::max(0.0f, x_data[i]);
    }
    
    return y;
}

// GELU activation (Gaussian Error Linear Unit)
Tensor gelu(const Tensor& x) {
    int64_t n = x.shape().numel();
    Tensor y = Tensor::empty(x.shape(), DType::F32);
    
    const float* x_data = x.data_f32();
    float* y_data = y.data_f32();
    
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    
    for (int64_t i = 0; i < n; ++i) {
        float xi = x_data[i];
        float tanh_arg = sqrt_2_over_pi * (xi + 0.044715f * xi * xi * xi);
        y_data[i] = 0.5f * xi * (1.0f + std::tanh(tanh_arg));
    }
    
    return y;
}

// SiLU (Swish) activation
Tensor silu(const Tensor& x) {
    int64_t n = x.shape().numel();
    Tensor y = Tensor::empty(x.shape(), DType::F32);
    
    const float* x_data = x.data_f32();
    float* y_data = y.data_f32();
    
    for (int64_t i = 0; i < n; ++i) {
        float xi = x_data[i];
        y_data[i] = xi / (1.0f + std::exp(-xi));
    }
    
    return y;
}

// Softmax
Tensor softmax(const Tensor& x) {
    // Apply softmax along last dimension
    int64_t n = x.shape().numel();
    
    if (x.shape().ndim() == 1) {
        // Simple 1D case
        Tensor y = Tensor::empty(x.shape(), DType::F32);
        const float* x_data = x.data_f32();
        float* y_data = y.data_f32();
        
        // Find max for numerical stability
        float max_val = x_data[0];
        for (int64_t i = 1; i < n; ++i) {
            max_val = std::max(max_val, x_data[i]);
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            y_data[i] = std::exp(x_data[i] - max_val);
            sum += y_data[i];
        }
        
        // Normalize
        for (int64_t i = 0; i < n; ++i) {
            y_data[i] /= sum;
        }
        
        return y;
    } else {
        // Multi-dimensional: apply along last axis
        // TODO: implement batched softmax
        throw std::runtime_error("Batched softmax not yet implemented");
    }
}

// RMSNorm
Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps) {
    int ndim = x.shape().ndim();
    
    if (ndim == 1) {
        // 1D case: [hidden_dim]
        int64_t n = x.shape().numel();
        
        if (weight.shape().numel() != n) {
            throw std::runtime_error("RMSNorm: weight size mismatch");
        }
        
        const float* x_data = x.data_f32();
        const float* w_data = weight.data_f32();
        
        // Compute RMS
        float rms = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            rms += x_data[i] * x_data[i];
        }
        rms = std::sqrt(rms / n + eps);
        
        // Normalize and scale
        Tensor y = Tensor::empty(x.shape(), DType::F32);
        float* y_data = y.data_f32();
        
        for (int64_t i = 0; i < n; ++i) {
            y_data[i] = (x_data[i] / rms) * w_data[i];
        }
        
        return y;
        
    } else if (ndim == 2) {
        // 2D case: [seq_len, hidden_dim] - normalize along last axis
        int64_t seq_len = x.shape().size(0);
        int64_t hidden_dim = x.shape().size(1);
        
        if (weight.shape().numel() != hidden_dim) {
            throw std::runtime_error("RMSNorm: weight size must match hidden_dim");
        }
        
        const float* x_data = x.data_f32();
        const float* w_data = weight.data_f32();
        
        Tensor y = Tensor::empty(x.shape(), DType::F32);
        float* y_data = y.data_f32();
        
        // Process each sequence position independently
        for (int64_t seq = 0; seq < seq_len; ++seq) {
            // Compute RMS for this sequence position
            float rms = 0.0f;
            int64_t offset = seq * hidden_dim;
            
            for (int64_t i = 0; i < hidden_dim; ++i) {
                float val = x_data[offset + i];
                rms += val * val;
            }
            rms = std::sqrt(rms / hidden_dim + eps);
            
            // Normalize and scale
            for (int64_t i = 0; i < hidden_dim; ++i) {
                y_data[offset + i] = (x_data[offset + i] / rms) * w_data[i];
            }
        }
        
        return y;
        
    } else {
        throw std::runtime_error("RMSNorm only supports 1D or 2D tensors");
    }
}

// Precompute RoPE frequencies
std::pair<Tensor, Tensor> precompute_rope_freqs(int max_seq_len, int head_dim, float theta) {
    if (head_dim % 2 != 0) {
        throw std::runtime_error("RoPE requires even head_dim");
    }
    
    int half_dim = head_dim / 2;
    
    Tensor freqs_cos = Tensor::empty({max_seq_len, half_dim}, DType::F32);
    Tensor freqs_sin = Tensor::empty({max_seq_len, half_dim}, DType::F32);
    
    float* cos_data = freqs_cos.data_f32();
    float* sin_data = freqs_sin.data_f32();
    
    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim);
            float angle = pos * freq;
            cos_data[pos * half_dim + i] = std::cos(angle);
            sin_data[pos * half_dim + i] = std::sin(angle);
        }
    }
    
    return {std::move(freqs_cos), std::move(freqs_sin)};
}

// RoPE application
Tensor rope(const Tensor& x, const Tensor& freqs_cos, const Tensor& freqs_sin) {
    // Simplified 2D implementation: [seq_len, head_dim]
    if (x.shape().ndim() != 2) {
        throw std::runtime_error("RoPE only supports 2D tensors for now");
    }
    
    int64_t seq_len = x.shape().size(0);
    int64_t head_dim = x.shape().size(1);
    int64_t half_dim = head_dim / 2;
    
    if (head_dim % 2 != 0) {
        throw std::runtime_error("RoPE requires even head_dim");
    }
    
    const float* x_data = x.data_f32();
    const float* cos_data = freqs_cos.data_f32();
    const float* sin_data = freqs_sin.data_f32();
    
    Tensor y = Tensor::empty(x.shape(), DType::F32);
    float* y_data = y.data_f32();
    
    for (int64_t pos = 0; pos < seq_len; ++pos) {
        for (int64_t i = 0; i < half_dim; ++i) {
            float x_even = x_data[pos * head_dim + 2*i];
            float x_odd = x_data[pos * head_dim + 2*i + 1];
            float cos_val = cos_data[pos * half_dim + i];
            float sin_val = sin_data[pos * half_dim + i];
            
            // Rotation: [cos, -sin; sin, cos] @ [x_even; x_odd]
            y_data[pos * head_dim + 2*i] = x_even * cos_val - x_odd * sin_val;
            y_data[pos * head_dim + 2*i + 1] = x_even * sin_val + x_odd * cos_val;
        }
    }
    
    return y;
}

// Attention scores (simplified)
Tensor attention_scores(const Tensor& q, const Tensor& k, float scale) {
    // Simplified: Q[seq_q, d], K[seq_k, d] -> scores[seq_q, seq_k]
    if (q.shape().ndim() != 2 || k.shape().ndim() != 2) {
        throw std::runtime_error("attention_scores: only 2D tensors supported for now");
    }
    
    int64_t seq_q = q.shape().size(0);
    int64_t d = q.shape().size(1);
    int64_t seq_k = k.shape().size(0);
    
    if (k.shape().size(1) != d) {
        throw std::runtime_error("attention_scores: Q and K must have same feature dim");
    }
    
    const float* q_data = q.data_f32();
    const float* k_data = k.data_f32();
    
    Tensor scores = Tensor::zeros({seq_q, seq_k}, DType::F32);
    float* scores_data = scores.data_f32();
    
    // Q @ K^T
    for (int64_t i = 0; i < seq_q; ++i) {
        for (int64_t j = 0; j < seq_k; ++j) {
            float dot = 0.0f;
            for (int64_t p = 0; p < d; ++p) {
                dot += q_data[i * d + p] * k_data[j * d + p];
            }
            scores_data[i * seq_k + j] = dot * scale;
        }
    }
    
    return scores;
}

// Apply attention mask
void apply_attention_mask(Tensor& scores, const Tensor& mask) {
    // mask: same shape as scores, 0 = keep, 1 = mask
    if (scores.shape().numel() != mask.shape().numel()) {
        throw std::runtime_error("apply_attention_mask: shape mismatch");
    }
    
    int64_t n = scores.shape().numel();
    float* scores_data = scores.data_f32();
    const float* mask_data = mask.data_f32();
    
    const float MASK_VALUE = -1e9f; // Large negative for softmax
    
    for (int64_t i = 0; i < n; ++i) {
        if (mask_data[i] > 0.5f) {
            scores_data[i] = MASK_VALUE;
        }
    }
}

// Expand KV for GQA
Tensor expand_kv_for_gqa(const Tensor& kv, int n_q_heads, int n_kv_heads) {
    if (n_q_heads % n_kv_heads != 0) {
        throw std::runtime_error("GQA: n_q_heads must be divisible by n_kv_heads");
    }
    
    int repeat = n_q_heads / n_kv_heads;
    
    if (repeat == 1) {
        // No expansion needed
        return Tensor::from_data(kv.data(), kv.shape(), kv.dtype());
    }
    
    // TODO: implement actual expansion
    throw std::runtime_error("GQA expansion not yet implemented");
}

// Utilities
namespace utils {

void copy(const Tensor& src, Tensor& dst) {
    if (src.shape().numel() != dst.shape().numel()) {
        throw std::runtime_error("copy: shape mismatch");
    }
    std::memcpy(dst.data(), src.data(), src.size_bytes());
}

void fill(Tensor& t, float value) {
    int64_t n = t.shape().numel();
    float* data = t.data_f32();
    for (int64_t i = 0; i < n; ++i) {
        data[i] = value;
    }
}

void print(const Tensor& t, const std::string& name) {
    std::stringstream ss;
    ss << name << " " << t.shape().to_string() << ": ";
    
    const float* data = t.data_f32();
    int64_t n = std::min<int64_t>(10, t.shape().numel());
    
    for (int64_t i = 0; i < n; ++i) {
        ss << std::fixed << std::setprecision(4) << data[i] << " ";
    }
    
    if (t.shape().numel() > n) {
        ss << "...";
    }
    
    Logger::instance().info(ss.str());
}

float norm(const Tensor& t) {
    const float* data = t.data_f32();
    int64_t n = t.shape().numel();
    
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i] * data[i];
    }
    
    return std::sqrt(sum);
}

bool allclose(const Tensor& a, const Tensor& b, float rtol, float atol) {
    if (a.shape().numel() != b.shape().numel()) {
        return false;
    }
    
    const float* a_data = a.data_f32();
    const float* b_data = b.data_f32();
    int64_t n = a.shape().numel();
    
    for (int64_t i = 0; i < n; ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        float threshold = atol + rtol * std::abs(b_data[i]);
        if (diff > threshold) {
            return false;
        }
    }
    
    return true;
}

} // namespace utils

} // namespace ash
