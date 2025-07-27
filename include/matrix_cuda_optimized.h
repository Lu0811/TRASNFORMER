#ifndef MATRIX_CUDA_OPTIMIZED_H
#define MATRIX_CUDA_OPTIMIZED_H

#include "matrix.h"
#include <memory>
#include <unordered_map>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

// ============================================================================
// OPTIMIZED CUDA MEMORY MANAGEMENT
// ============================================================================

class CudaMemoryPool {
private:
    static CudaMemoryPool* instance;
    std::unordered_map<size_t, std::vector<float*>> free_buffers;
    std::unordered_map<float*, size_t> allocated_sizes;
    size_t total_allocated;
    size_t peak_usage;
    
    CudaMemoryPool();
    ~CudaMemoryPool();
    
public:
    static CudaMemoryPool* getInstance();
    
    float* allocate(size_t size);
    void deallocate(float* ptr);
    void clear();
    
    size_t getTotalAllocated() const { return total_allocated; }
    size_t getPeakUsage() const { return peak_usage; }
};

// ============================================================================
// CUDA CONTEXT MANAGER
// ============================================================================

class CudaContext {
private:
    static CudaContext* instance;
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
    bool initialized;
    
    CudaContext();
    ~CudaContext();
    
public:
    static CudaContext* getInstance();
    
    cublasHandle_t getCublasHandle() { return cublas_handle; }
    cudnnHandle_t getCudnnHandle() { return cudnn_handle; }
    
    bool isInitialized() const { return initialized; }
    void warmup();
};

// ============================================================================
// OPTIMIZED CUDA MATRIX CLASS
// ============================================================================

class CudaMatrix {
private:
    float* d_data;          // GPU memory pointer
    float* h_data;          // CPU memory pointer (for synchronization)
    int rows, cols;
    bool data_on_gpu;       // Track where current data is
    bool data_on_cpu;       // Track where current data is
    bool owns_gpu_memory;   // Track if we need to free GPU memory
    
    CudaMemoryPool* memory_pool;
    
public:
    // Constructors
    CudaMatrix(int rows, int cols);
    CudaMatrix(const Matrix& cpu_matrix);
    CudaMatrix(const CudaMatrix& other);
    CudaMatrix& operator=(const CudaMatrix& other);
    ~CudaMatrix();
    
    // Memory management
    void ensureGpuData();
    void ensureCpuData();
    void syncToGpu();
    void syncToCpu();
    Matrix toCpuMatrix() const;
    
    // Basic properties
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getSize() const { return rows * cols; }
    float* getGpuData() { ensureGpuData(); return d_data; }
    const float* getGpuData() const { return d_data; }
    
    // CUDA operations (optimized)
    CudaMatrix cudaAdd(const CudaMatrix& other) const;
    CudaMatrix cudaSub(const CudaMatrix& other) const;
    CudaMatrix cudaMultiply(const CudaMatrix& other) const;
    CudaMatrix cudaMultiply(float scalar) const;
    CudaMatrix cudaTranspose() const;
    CudaMatrix cudaSoftmax() const;
    CudaMatrix cudaRelu() const;
    CudaMatrix cudaGelu() const;
    CudaMatrix cudaLayerNorm(const CudaMatrix& gamma, const CudaMatrix& beta, float eps = 1e-6) const;
    
    // Advanced operations
    CudaMatrix cudaDropout(float rate, bool training = true) const;
    std::pair<CudaMatrix, CudaMatrix> cudaLayerNormWithStats(const CudaMatrix& gamma, const CudaMatrix& beta, float eps = 1e-6) const;
    
    // Slice operations
    CudaMatrix slice(int start_row, int end_row, int start_col, int end_col) const;
    void setSlice(int start_row, int start_col, const CudaMatrix& other);
    
    // Reduction operations
    float cudaSum() const;
    float cudaMax() const;
    float cudaMean() const;
    CudaMatrix cudaRowSum() const;
    CudaMatrix cudaColSum() const;
    
    // In-place operations
    void cudaAddInplace(const CudaMatrix& other);
    void cudaSubInplace(const CudaMatrix& other);
    void cudaMultiplyInplace(float scalar);
    void cudaZero();
    
    // Utilities
    void print(const std::string& name = "") const;
    bool isValid() const;
};

// ============================================================================
// CUDA ATTENTION IMPLEMENTATION
// ============================================================================

class CudaMultiHeadAttention {
private:
    int d_model, num_heads, d_k;
    CudaMatrix W_q, W_k, W_v, W_o;
    CudaMatrix b_q, b_k, b_v, b_o;
    
    // Temporary buffers for efficiency
    mutable std::unique_ptr<CudaMatrix> temp_q, temp_k, temp_v;
    mutable std::unique_ptr<CudaMatrix> temp_scores, temp_weights;
    
public:
    CudaMultiHeadAttention(int d_model, int num_heads);
    
    CudaMatrix forward(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, bool mask = false);
    std::tuple<CudaMatrix, CudaMatrix, CudaMatrix> backward(const CudaMatrix& grad_output);
    
    void updateWeights(const CudaMatrix& grad_q, const CudaMatrix& grad_k, 
                      const CudaMatrix& grad_v, const CudaMatrix& grad_o, 
                      float learning_rate);
    
    void zeroGradients();
    
private:
    CudaMatrix scaledDotProductAttention(const CudaMatrix& Q, const CudaMatrix& K, const CudaMatrix& V, bool mask = false) const;
    void reshapeForHeads(const CudaMatrix& input, CudaMatrix& output, bool transpose = false) const;
};

// ============================================================================
// CUDA BATCH PROCESSING
// ============================================================================

class CudaBatchProcessor {
private:
    int batch_size;
    int max_seq_len;
    int d_model;
    
    // Pre-allocated batch buffers
    std::unique_ptr<CudaMatrix> batch_input;
    std::unique_ptr<CudaMatrix> batch_output;
    std::unique_ptr<CudaMatrix> batch_gradients;
    
public:
    CudaBatchProcessor(int batch_size, int max_seq_len, int d_model);
    
    CudaMatrix processBatch(const std::vector<CudaMatrix>& inputs);
    std::vector<CudaMatrix> backwardBatch(const CudaMatrix& grad_output);
    
    void setBatchSize(int new_batch_size);
};

// ============================================================================
// KERNEL DECLARATIONS
// ============================================================================

extern "C" {
    // Optimized kernels
    void cuda_fused_attention_forward(const float* Q, const float* K, const float* V, 
                                     float* output, float* attention_weights,
                                     int batch_size, int seq_len, int num_heads, int d_k,
                                     float scale, bool mask);
    
    void cuda_fused_attention_backward(const float* grad_output, const float* Q, const float* K, const float* V,
                                      const float* attention_weights, float* grad_Q, float* grad_K, float* grad_V,
                                      int batch_size, int seq_len, int num_heads, int d_k, float scale);
    
    void cuda_layer_norm_forward(const float* input, const float* gamma, const float* beta,
                                float* output, float* mean, float* rstd,
                                int batch_size, int features, float eps);
    
    void cuda_layer_norm_backward(const float* grad_output, const float* input, const float* mean, const float* rstd,
                                 const float* gamma, float* grad_input, float* grad_gamma, float* grad_beta,
                                 int batch_size, int features);
    
    void cuda_gelu_forward(const float* input, float* output, int size);
    void cuda_gelu_backward(const float* grad_output, const float* input, float* grad_input, int size);
    
    void cuda_dropout_forward(const float* input, float* output, float* mask, 
                             int size, float rate, bool training, unsigned long long seed);
    void cuda_dropout_backward(const float* grad_output, const float* mask, float* grad_input, 
                              int size, float rate);
    
    void cuda_softmax_forward(const float* input, float* output, int batch_size, int features);
    void cuda_softmax_backward(const float* grad_output, const float* softmax_output, 
                              float* grad_input, int batch_size, int features);
    
    void cuda_cross_entropy_loss(const float* predictions, const int* labels, 
                                 float* loss, float* grad_predictions,
                                 int batch_size, int num_classes);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error("CUDA Error"); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS Error at " << __FILE__ << ":" << __LINE__ \
                      << " - Status: " << status << std::endl; \
            throw std::runtime_error("CUBLAS Error"); \
        } \
    } while(0)

// Performance utilities
void cudaProfilerStart();
void cudaProfilerStop();
void cudaWarmup();
void cudaPrintMemoryUsage();

// Conversion utilities
CudaMatrix convertToCuda(const Matrix& cpu_matrix);
Matrix convertToCpu(const CudaMatrix& cuda_matrix);
std::vector<CudaMatrix> convertBatchToCuda(const std::vector<Matrix>& cpu_batch);
std::vector<Matrix> convertBatchToCpu(const std::vector<CudaMatrix>& cuda_batch);

#endif // USE_CUDA

#endif // MATRIX_CUDA_OPTIMIZED_H