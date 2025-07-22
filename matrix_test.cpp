#include "../include/matrix.h"
#include <iostream>
#include <chrono>

void test_add_scalar() {
    Matrix A(2, 2); A.data = {{1,2},{3,4}};
    Matrix B(1, 1); B.data = {{5}};
    Matrix C = A + B;
    std::cout << "test_add_scalar: ";
    if (C.data[0][0] == 6 && C.data[1][1] == 9) std::cout << "OK\n";
    else std::cout << "FAIL\n";
}

void test_add_row() {
    Matrix A(2, 3); A.data = {{1,2,3},{4,5,6}};
    Matrix B(1, 3); B.data = {{10,20,30}};
    Matrix C = A + B;
    std::cout << "test_add_row: ";
    if (C.data[0][1] == 22 && C.data[1][2] == 36) std::cout << "OK\n";
    else std::cout << "FAIL\n";
}

void test_add_col() {
    Matrix A(2, 3); A.data = {{1,2,3},{4,5,6}};
    Matrix B(2, 1); B.data = {{100},{200}};
    Matrix C = A + B;
    std::cout << "test_add_col: ";
    if (C.data[0][0] == 101 && C.data[1][2] == 206) std::cout << "OK\n";
    else std::cout << "FAIL\n";
}

void test_matmul_scalar() {
    Matrix A(28, 28);
    // Inicializa A con valores predecibles
    for (int i = 0; i < 28; ++i)
        for (int j = 0; j < 28; ++j)
            A.data[i][j] = i + j;
    Matrix C = A.cudaMultiply(2.0f);
    std::cout << "test_matmul_scalar: ";
    double tol = 1e-6;
    bool ok = true;
    // Verifica y muestra algunos valores para diagnóstico
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (std::fabs(C.data[i][j] - (A.data[i][j]*2)) > tol) ok = false;
            if (i < 2 && j < 2) {
                std::cout << "A[" << i << "][" << j << "]=" << A.data[i][j]
                          << ", C[" << i << "][" << j << "]=" << C.data[i][j] << "; ";
            }
        }
    }
    std::cout << (ok ? "OK" : "FAIL") << std::endl;
}

void test_matmul_incompatible() {
    Matrix A(2,3); A.data = {{1,2,3},{4,5,6}};
    Matrix B(2,2); B.data = {{1,2},{3,4}};
    std::cout << "test_matmul_incompatible: ";
    try {
        Matrix C = A.cudaMultiply(B);
        std::cout << "FAIL\n";
    } catch(const std::exception& e) {
        std::cout << "OK\n";
    }
}

void test_matmul_timing() {
    Matrix A(128,128); Matrix B(128,128);
    for(int i=0;i<128;i++) for(int j=0;j<128;j++) {A.data[i][j]=1; B.data[i][j]=2;}
    auto start = std::chrono::high_resolution_clock::now();
    Matrix C = A.cudaMultiply(B);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end-start;
    std::cout << "test_matmul_timing: " << ms.count() << " ms\n";
}

int main() {
    test_add_scalar();
    test_add_row();
    test_add_col();
    test_matmul_scalar();
    test_matmul_incompatible();
    test_matmul_timing();
    return 0;
}
