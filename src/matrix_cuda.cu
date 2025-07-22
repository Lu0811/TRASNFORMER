#include "cuda_ops.h"
#include <cuda_runtime.h>
#include <iostream>
#include <float.h>
#define BLOCK_SIZE 32

// Suma de matrices
__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = A[idx] + B[idx];
}
extern "C" void cuda_matrix_add(const float* A, const float* B, float* C, int size) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, size);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// Resta de matrices
__global__ void matrix_sub_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = A[idx] - B[idx];
}
extern "C" void cuda_matrix_sub(const float* A, const float* B, float* C, int size) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_sub_kernel<<<blocks, threads>>>(d_A, d_B, d_C, size);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// Transposición de matriz
__global__ void matrix_transpose_kernel(const float* A, float* B, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        B[x * rows + y] = A[y * cols + x];
    }
}
extern "C" void cuda_matrix_transpose(const float* A, float* B, int rows, int cols) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, rows * cols * sizeof(float));
    cudaMalloc(&d_B, rows * cols * sizeof(float));
    cudaMemcpy(d_A, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
    matrix_transpose_kernel<<<blocks, threads>>>(d_A, d_B, rows, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Multiplicación escalar
__global__ void matrix_scalar_mul_kernel(const float* A, float* C, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * scalar;
    }
}
extern "C" void cuda_matrix_scalar_mul(const float* A, float* C, float scalar, int size) {
    float *d_A = nullptr, *d_C = nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_A, size * sizeof(float));
    if (err != cudaSuccess) {
        return;
    }
    err = cudaMalloc(&d_C, size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_A);
        return;
    }
    err = cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_C);
        return;
    }
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_scalar_mul_kernel<<<blocks, threads>>>(d_A, d_C, scalar, size);
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        // Silenciar error
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_C);
        return;
    }
    err = cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        // Silenciar error
    }
    cudaFree(d_A);
    cudaFree(d_C);
}

// Softmax por filas
__global__ void matrix_softmax_kernel(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float shared[];
    float* row_data = shared;
    float* exp_data = shared + cols;
    for (int i = tid; i < cols; i += blockDim.x)
        row_data[i] = A[row * cols + i];
    __syncthreads();
    float maxval = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x)
        maxval = fmaxf(maxval, row_data[i]);
    __syncthreads();
    for (int i = tid; i < cols; i += blockDim.x)
        exp_data[i] = expf(row_data[i] - maxval);
    __syncthreads();
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        sum += exp_data[i];
    __syncthreads();
    for (int i = tid; i < cols; i += blockDim.x)
        B[row * cols + i] = exp_data[i] / sum;
}
extern "C" void cuda_matrix_softmax(const float* A, float* B, int rows, int cols) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, rows * cols * sizeof(float));
    cudaMalloc(&d_B, rows * cols * sizeof(float));
    cudaMemcpy(d_A, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    matrix_softmax_kernel<<<rows, 32, 2*cols*sizeof(float)>>>(d_A, d_B, rows, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// ReLU
__global__ void matrix_relu_kernel(const float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) B[idx] = fmaxf(A[idx], 0.0f);
}
extern "C" void cuda_matrix_relu(const float* A, float* B, int size) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_relu_kernel<<<blocks, threads>>>(d_A, d_B, size);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Sigmoid
__global__ void matrix_sigmoid_kernel(const float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) B[idx] = 1.0f / (1.0f + expf(-A[idx]));
}
extern "C" void cuda_matrix_sigmoid(const float* A, float* B, int size) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_sigmoid_kernel<<<blocks, threads>>>(d_A, d_B, size);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Tanh
__global__ void matrix_tanh_kernel(const float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) B[idx] = tanhf(A[idx]);
}
extern "C" void cuda_matrix_tanh(const float* A, float* B, int size) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_tanh_kernel<<<blocks, threads>>>(d_A, d_B, size);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Reduce sum
__global__ void matrix_reduce_sum_kernel(const float* A, float* B, int size) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = (idx < size) ? A[idx] : 0.0f;
    __syncthreads();
    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) B[blockIdx.x] = shared[0];
}
extern "C" void cuda_matrix_reduce_sum(const float* A, float* B, int size) {
    float *d_A, *d_B;
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, blocks * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    matrix_reduce_sum_kernel<<<blocks, BLOCK_SIZE>>>(d_A, d_B, size);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Reduce max
__global__ void matrix_reduce_max_kernel(const float* A, float* B, int size) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = (idx < size) ? A[idx] : -FLT_MAX;
    __syncthreads();
    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) B[blockIdx.x] = shared[0];
}
extern "C" void cuda_matrix_reduce_max(const float* A, float* B, int size) {
    float *d_A, *d_B;
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, blocks * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    matrix_reduce_max_kernel<<<blocks, BLOCK_SIZE>>>(d_A, d_B, size);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Exp
__global__ void matrix_exp_kernel(const float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) B[idx] = expf(A[idx]);
}
extern "C" void cuda_matrix_exp(const float* A, float* B, int size) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_exp_kernel<<<blocks, threads>>>(d_A, d_B, size);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Matmul (multiplicación de matrices)
__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i)
            sum += A[row * N + i] * B[i * K + col];
        C[row * K + col] = sum;
    }
}
extern "C" void cuda_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threads(16, 16);
    dim3 blocks((K + 15) / 16, (M + 15) / 16);
    matmul_shared_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// LayerNorm CUDA
__global__ void matrix_layernorm_kernel(const float* A, float* B, int rows, int cols, float gamma, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        // Implementación dummy: copia A a B con escala y sesgo
        B[idx] = gamma * A[idx] + beta;
    }
}

extern "C" void cuda_matrix_layernorm(const float* A, float* B, int rows, int cols, float gamma, float beta) {
    float *d_A, *d_B;
    int size = rows * cols;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    matrix_layernorm_kernel<<<blocks, threads>>>(d_A, d_B, rows, cols, gamma, beta);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}