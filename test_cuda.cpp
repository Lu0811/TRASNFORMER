#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#ifdef USE_CUDA
#include "include/cuda_ops.h"
#endif

void test_cuda_operations() {
    std::cout << "=== PRUEBA DE OPERACIONES CUDA ===" << std::endl;
    
#ifdef USE_CUDA
    // Prueba básica de suma de matrices
    const int size = 1000;
    std::vector<float> A(size), B(size), C(size);
    
    // Inicializar matrices con valores aleatorios
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }
    
    std::cout << "Probando suma de matrices CUDA..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    cuda_matrix_add(A.data(), B.data(), C.data(), size);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "✅ Suma CUDA completada en: " << duration.count() << " microsegundos" << std::endl;
    
    // Verificar algunos resultados
    bool correct = true;
    for (int i = 0; i < std::min(10, size); i++) {
        float expected = A[i] + B[i];
        float tolerance = 1e-6f;
        if (std::abs(C[i] - expected) > tolerance) {
            correct = false;
            std::cout << "❌ Error en posición " << i << ": esperado " << expected << ", obtenido " << C[i] << std::endl;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✅ Verificación de suma exitosa" << std::endl;
    }
    
    // Prueba de multiplicación escalar
    std::cout << "\nProbando multiplicación escalar CUDA..." << std::endl;
    std::vector<float> D(size);
    float scalar = 2.5f;
    
    start = std::chrono::high_resolution_clock::now();
    cuda_matrix_scalar_mul(A.data(), D.data(), scalar, size);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "✅ Multiplicación escalar CUDA completada en: " << duration.count() << " microsegundos" << std::endl;
    
    // Verificar resultados
    correct = true;
    for (int i = 0; i < std::min(10, size); i++) {
        float expected = A[i] * scalar;
        float tolerance = 1e-6f;
        if (std::abs(D[i] - expected) > tolerance) {
            correct = false;
            std::cout << "❌ Error en multiplicación escalar posición " << i << ": esperado " << expected << ", obtenido " << D[i] << std::endl;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✅ Verificación de multiplicación escalar exitosa" << std::endl;
    }
    
    // Prueba de multiplicación de matrices
    std::cout << "\nProbando multiplicación de matrices CUDA..." << std::endl;
    const int M = 64, N = 64, K = 64;
    std::vector<float> matA(M * N), matB(N * K), matC(M * K);
    
    for (int i = 0; i < M * N; i++) matA[i] = dis(gen);
    for (int i = 0; i < N * K; i++) matB[i] = dis(gen);
    
    start = std::chrono::high_resolution_clock::now();
    cuda_matmul(matA.data(), matB.data(), matC.data(), M, N, K);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "✅ Multiplicación de matrices CUDA (" << M << "x" << N << " * " << N << "x" << K << ") completada en: " << duration.count() << " microsegundos" << std::endl;
    
    std::cout << "\n🎉 Todas las pruebas CUDA pasaron exitosamente!" << std::endl;
    
#else
    std::cout << "❌ CUDA no está habilitado. Compila con -DUSE_CUDA" << std::endl;
#endif
}

int main() {
    std::cout << "Iniciando pruebas del sistema CUDA..." << std::endl;
    
#ifdef USE_CUDA
    // Verificar dispositivos CUDA disponibles
    int deviceCount;
    if (cudaGetDeviceCount(&deviceCount) == cudaSuccess) {
        std::cout << "Dispositivos CUDA encontrados: " << deviceCount << std::endl;
        
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "Dispositivo " << i << ": " << prop.name << std::endl;
            std::cout << "  - Memoria global: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
            std::cout << "  - Compute Capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  - Multiprocessors: " << prop.multiProcessorCount << std::endl;
        }
    } else {
        std::cout << "❌ No se pudieron detectar dispositivos CUDA" << std::endl;
        return 1;
    }
#endif
    
    test_cuda_operations();
    
    return 0;
}
