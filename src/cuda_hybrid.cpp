// CUDA Hybrid Implementation 
#include "../include/cuda_ops.h" 
#include <cuda_runtime.h> 
#include <cublas_v2.h> 
#include <iostream> 
#include <vector> 
 
// Variables globales CUBLAS 
static cublasHandle_t handle = nullptr; 
static bool cublas_init = false; 
 
void init_cublas() { 
    if (!cublas_init) { 
        cublasCreate(&handle); 
        cublas_init = true; 
        std::cout << "✅ CUBLAS inicializado" << std::endl; 
    } 
} 
 
extern "C" void cuda_matmul(const float* A, const float* B, float* C, int M, int N, int K) { 
    init_cublas(); 
    float *d_A, *d_B, *d_C; 
    cudaMalloc(&d_A, M * N * sizeof(float)); 
    cudaMalloc(&d_B, N * K * sizeof(float)); 
    cudaMalloc(&d_C, M * K * sizeof(float)); 
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice); 
    const float alpha = 1.0f, beta = 0.0f; 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, d_B, K, d_A, N, &beta, d_C, K); 
    cudaMemcpy(C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost); 
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); 
} 
 
extern "C" void cuda_matrix_add(const float* A, const float* B, float* C, int size) { 
    for (int i = 0; i < size; i++) C[i] = A[i] + B[i]; 
} 
 
extern "C" void cuda_matrix_sub(const float* A, const float* B, float* C, int size) { 
    for (int i = 0; i < size; i++) C[i] = A[i] - B[i]; 
} 
 
extern "C" void cuda_matrix_scalar_mul(const float* A, float* C, float scalar, int size) { 
    for (int i = 0; i < size; i++) C[i] = A[i] * scalar; 
} 
