#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16

// Kernel optimizado con memoria compartida
__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < M && t * BLOCK_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < K && t * BLOCK_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * K + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int i = 0; i < BLOCK_SIZE; ++i)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < K)
        C[row * K + col] = sum;
}

inline void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

extern "C" {
    void cuda_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
        std::cout << "[CUDA] Llamando a cuda_matmul (GPU)" << std::endl;
        size_t size_A = M * N * sizeof(float);
        size_t size_B = N * K * sizeof(float);
        size_t size_C = M * K * sizeof(float);
        float *d_A, *d_B, *d_C;
        checkCuda(cudaMalloc(&d_A, size_A), "cudaMalloc d_A");
        checkCuda(cudaMalloc(&d_B, size_B), "cudaMalloc d_B");
        checkCuda(cudaMalloc(&d_C, size_C), "cudaMalloc d_C");
        checkCuda(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice), "cudaMemcpy A");
        checkCuda(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice), "cudaMemcpy B");
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul_shared_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        checkCuda(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost), "cudaMemcpy C");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    // Ejemplo de nueva función CUDA: suma de matrices
    __global__ void matrix_add_kernel(const float* A, const float* B, float* C, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = A[idx] + B[idx];
        }
    }

    void cuda_matrix_add(const float* A, const float* B, float* C, int size) {
        std::cout << "[CUDA] Llamando a cuda_matrix_add (GPU)" << std::endl;
        float *d_A, *d_B, *d_C;
        checkCuda(cudaMalloc(&d_A, size * sizeof(float)), "cudaMalloc d_A");
        checkCuda(cudaMalloc(&d_B, size * sizeof(float)), "cudaMalloc d_B");
        checkCuda(cudaMalloc(&d_C, size * sizeof(float)), "cudaMalloc d_C");
        checkCuda(cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy A");
        checkCuda(cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy B");
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        matrix_add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, size);
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        checkCuda(cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy C");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}
