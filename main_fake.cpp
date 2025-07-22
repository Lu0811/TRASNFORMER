#include <iostream>
#include <chrono>
#include "matrix.h"

int main() {
    const int SIZE = 512;

    Matrix A(SIZE, SIZE);
    Matrix B(SIZE, SIZE);

    A.randomize(-1.0, 1.0);
    B.randomize(-1.0, 1.0);

    // 🧠 Multiplicación con CUDA
    auto start_gpu = std::chrono::high_resolution_clock::now();
    Matrix C_gpu = A.cudaMultiply(B);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double ms_gpu = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
    std::cout << "⏱️ Tiempo CUDA (GPU): " << ms_gpu << " ms" << std::endl;

    // 🧠 Multiplicación clásica en CPU (opcional)
    auto start_cpu = std::chrono::high_resolution_clock::now();
    Matrix C_cpu = A * B;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "⏱️ Tiempo CPU: " << ms_cpu << " ms" << std::endl;

    return 0;
}
