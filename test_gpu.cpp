#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "include/matrix.h"
#include <chrono>

void test_gpu_performance() {
    std::cout << "\n🔍 === TEST DE RENDIMIENTO GPU ===" << std::endl;
    
    // Verificar GPU
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cout << "❌ No se detectó GPU CUDA" << std::endl;
        return;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "✅ GPU detectada: " << prop.name << std::endl;
    std::cout << "   Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "   Memoria: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    // Test de multiplicación matricial
    std::cout << "\n📊 Test de multiplicación matricial:" << std::endl;
    
    // Matrices pequeñas (256x256)
    Matrix A(256, 256), B(256, 256);
    A.randomize(); B.randomize();
    
    auto start = std::chrono::high_resolution_clock::now();
    Matrix C = A.cudaMultiply(B);
    auto end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "   GPU (256x256): " << gpu_time << " μs" << std::endl;
    
    // Matrices medianas (512x512)
    Matrix A2(512, 512), B2(512, 512);
    A2.randomize(); B2.randomize();
    
    start = std::chrono::high_resolution_clock::now();
    Matrix C2 = A2.cudaMultiply(B2);
    end = std::chrono::high_resolution_clock::now();
    gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "   GPU (512x512): " << gpu_time << " ms" << std::endl;
    
    // Test de operaciones Transformer
    std::cout << "\n📊 Test de operaciones Transformer:" << std::endl;
    
    // Softmax
    Matrix scores(49, 49);  // 49 patches
    scores.randomize(-1, 1);
    
    start = std::chrono::high_resolution_clock::now();
    Matrix attn = scores.cudaSoftmax();
    end = std::chrono::high_resolution_clock::now();
    auto softmax_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "   Softmax (49x49): " << softmax_time << " μs" << std::endl;
    
    // GELU
    Matrix hidden(49, 512);  // Feed-forward
    hidden.randomize();
    
    start = std::chrono::high_resolution_clock::now();
    Matrix activated = hidden.cudaGelu();
    end = std::chrono::high_resolution_clock::now();
    auto gelu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "   GELU (49x512): " << gelu_time << " μs" << std::endl;
    
    // Dropout
    start = std::chrono::high_resolution_clock::now();
    Matrix dropped = activated.cudaDropout(0.15, 42);
    end = std::chrono::high_resolution_clock::now();
    auto dropout_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "   Dropout (49x512): " << dropout_time << " μs" << std::endl;
    
    // Verificar memoria GPU
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "\n💾 Memoria GPU:" << std::endl;
    std::cout << "   Total: " << total_mem / (1024*1024) << " MB" << std::endl;
    std::cout << "   Libre: " << free_mem / (1024*1024) << " MB" << std::endl;
    std::cout << "   Usada: " << (total_mem - free_mem) / (1024*1024) << " MB" << std::endl;
    
    std::cout << "\n✅ GPU funcionando correctamente para Transformer" << std::endl;
}

int main() {
    std::cout << "🚀 VERIFICADOR DE GPU PARA TRANSFORMER" << std::endl;
    
    try {
        test_gpu_performance();
    } catch (const std::exception& e) {
        std::cout << "❌ Error: " << e.what() << std::endl;
    }
    
    return 0;
}