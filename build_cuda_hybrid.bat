@echo off
echo ========================================
echo  SOLUCION CUDA HIBRIDA
echo  RTX 3050 + Runtime CUDA + CPU Kernels
echo ========================================

echo 🎯 Tu GPU: RTX 3050 Laptop (Compute 8.6)
echo 🔧 CUDA Runtime: 12.8
echo 💡 Estrategia: Usar CUDA Runtime sin compilar kernels

REM Limpiar compilaciones anteriores
if exist "obj" rmdir /s /q "obj"
if exist "TransformerCUDA_Hybrid.exe" del "TransformerCUDA_Hybrid.exe"
mkdir obj

echo.
echo 📝 Creando wrapper CUDA híbrido...

REM Crear archivo híbrido que use CUDA Runtime pero kernels CPU optimizados
echo // CUDA Hybrid Implementation > src\cuda_hybrid.cpp
echo #include ^"../include/cuda_ops.h^" >> src\cuda_hybrid.cpp
echo #include ^<cuda_runtime.h^> >> src\cuda_hybrid.cpp
echo #include ^<cublas_v2.h^> >> src\cuda_hybrid.cpp
echo #include ^<iostream^> >> src\cuda_hybrid.cpp
echo #include ^<vector^> >> src\cuda_hybrid.cpp
echo. >> src\cuda_hybrid.cpp
echo // Variables globales CUBLAS >> src\cuda_hybrid.cpp
echo static cublasHandle_t handle = nullptr; >> src\cuda_hybrid.cpp
echo static bool cublas_init = false; >> src\cuda_hybrid.cpp
echo. >> src\cuda_hybrid.cpp
echo void init_cublas^(^) { >> src\cuda_hybrid.cpp
echo     if ^(!cublas_init^) { >> src\cuda_hybrid.cpp
echo         cublasCreate^(^&handle^); >> src\cuda_hybrid.cpp
echo         cublas_init = true; >> src\cuda_hybrid.cpp
echo         std::cout ^<^< ^"✅ CUBLAS inicializado^" ^<^< std::endl; >> src\cuda_hybrid.cpp
echo     } >> src\cuda_hybrid.cpp
echo } >> src\cuda_hybrid.cpp
echo. >> src\cuda_hybrid.cpp
echo extern ^"C^" void cuda_matmul^(const float* A, const float* B, float* C, int M, int N, int K^) { >> src\cuda_hybrid.cpp
echo     init_cublas^(^); >> src\cuda_hybrid.cpp
echo     float *d_A, *d_B, *d_C; >> src\cuda_hybrid.cpp
echo     cudaMalloc^(^&d_A, M * N * sizeof^(float^)^); >> src\cuda_hybrid.cpp
echo     cudaMalloc^(^&d_B, N * K * sizeof^(float^)^); >> src\cuda_hybrid.cpp
echo     cudaMalloc^(^&d_C, M * K * sizeof^(float^)^); >> src\cuda_hybrid.cpp
echo     cudaMemcpy^(d_A, A, M * N * sizeof^(float^), cudaMemcpyHostToDevice^); >> src\cuda_hybrid.cpp
echo     cudaMemcpy^(d_B, B, N * K * sizeof^(float^), cudaMemcpyHostToDevice^); >> src\cuda_hybrid.cpp
echo     const float alpha = 1.0f, beta = 0.0f; >> src\cuda_hybrid.cpp
echo     cublasSgemm^(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, ^&alpha, d_B, K, d_A, N, ^&beta, d_C, K^); >> src\cuda_hybrid.cpp
echo     cudaMemcpy^(C, d_C, M * K * sizeof^(float^), cudaMemcpyDeviceToHost^); >> src\cuda_hybrid.cpp
echo     cudaFree^(d_A^); cudaFree^(d_B^); cudaFree^(d_C^); >> src\cuda_hybrid.cpp
echo } >> src\cuda_hybrid.cpp
echo. >> src\cuda_hybrid.cpp
echo extern ^"C^" void cuda_matrix_add^(const float* A, const float* B, float* C, int size^) { >> src\cuda_hybrid.cpp
echo     for ^(int i = 0; i ^< size; i++^) C[i] = A[i] + B[i]; >> src\cuda_hybrid.cpp
echo } >> src\cuda_hybrid.cpp
echo. >> src\cuda_hybrid.cpp
echo extern ^"C^" void cuda_matrix_sub^(const float* A, const float* B, float* C, int size^) { >> src\cuda_hybrid.cpp
echo     for ^(int i = 0; i ^< size; i++^) C[i] = A[i] - B[i]; >> src\cuda_hybrid.cpp
echo } >> src\cuda_hybrid.cpp
echo. >> src\cuda_hybrid.cpp
echo extern ^"C^" void cuda_matrix_scalar_mul^(const float* A, float* C, float scalar, int size^) { >> src\cuda_hybrid.cpp
echo     for ^(int i = 0; i ^< size; i++^) C[i] = A[i] * scalar; >> src\cuda_hybrid.cpp
echo } >> src\cuda_hybrid.cpp

echo ✅ Wrapper híbrido creado

echo.
echo 🔨 Compilando implementación híbrida CUDA...
g++ -std=c++17 -O3 -DUSE_CUDA ^
    -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
    -Iinclude ^
    -c src/cuda_hybrid.cpp -o obj/cuda_hybrid.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando wrapper CUDA
    pause
    exit /b 1
)

echo ✅ Wrapper CUDA compilado

echo.
echo 🔨 Compilando archivos del proyecto...
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude -c src/matrix.cpp -o obj/matrix.o
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude -c src/mnist_loader.cpp -o obj/mnist_loader.o
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude -c src/transformer.cpp -o obj/transformer.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando archivos del proyecto
    pause
    exit /b 1
)

echo ✅ Archivos del proyecto compilados

echo.
echo 🔗 Enlazando con CUDA Runtime + CUBLAS...
g++ -std=c++17 -O3 -DUSE_CUDA ^
    -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
    -Iinclude ^
    -o TransformerCUDA_Hybrid.exe main_fast.cpp ^
    obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/cuda_hybrid.o ^
    -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
    -lcudart -lcublas

if %ERRORLEVEL% neq 0 (
    echo ⚠️  Enlazado con ruta completa falló, probando simple...
    g++ -std=c++17 -O3 -DUSE_CUDA ^
        -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
        -Iinclude ^
        -o TransformerCUDA_Hybrid.exe main_fast.cpp ^
        obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/cuda_hybrid.o ^
        -lcudart -lcublas
        
    if %ERRORLEVEL% neq 0 (
        echo ❌ Error en enlazado
        pause
        exit /b 1
    )
)

echo.
echo 🎉 ¡TRANSFORMER CUDA HÍBRIDO COMPILADO!
echo.
echo 🚀 Ejecutable: TransformerCUDA_Hybrid.exe
echo 🎯 GPU: RTX 3050 Laptop (Ampere 8.6)
echo ⚡ Aceleración: CUDA Runtime + CUBLAS para matrices
echo 🔧 Kernels: CPU optimizado donde sea necesario
echo 📊 Configuración: 2 épocas, batch 256, subset rápido
echo.
echo Para ejecutar:
echo   .\TransformerCUDA_Hybrid.exe
echo.
pause
