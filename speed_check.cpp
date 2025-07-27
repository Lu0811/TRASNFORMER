#include <iostream> 
#include <cuda_runtime.h> 
#include <cublas_v2.h> 
#include <chrono> 
int main() { 
    printf("⚡ VERIFICACIÓN VELOCIDAD MÁXIMA\\n\\n"); 
    printf("GPU: %s ^(%zu MB^)\\n", prop.name, prop.totalGlobalMem/1024/1024); 
    // Test velocidad pequeña 
    const int N = 128; 
    float *d_A, *d_B, *d_C; 
    float alpha=1.0f, beta=0.0f; 
    auto start = std::chrono::high_resolution_clock::now(); 
    for(int i=0; i<100; i++) { 
    } 
    cudaDeviceSynchronize(); 
    auto end = std::chrono::high_resolution_clock::now(); 
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count(); 
    printf("\\n⚡ 100x multiplicaciones 128x128: %ld μs\\n", time); 
    printf("⚡ Promedio por operación: %ld μs\\n", time/100); 
    if(time/100 < 100) printf("✅ GPU ULTRA-RÁPIDA\\n"); 
    cublasDestroy(handle); 
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); 
    return 0; 
} 
