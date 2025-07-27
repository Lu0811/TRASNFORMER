@echo off
echo ===================================
echo  Compilador CUDA Real para Transformer
echo  Optimizado para entrenamiento rápido
echo ===================================

REM Limpiar compilaciones anteriores
if exist "obj" rmdir /s /q "obj"
if exist "*.exe" del /q "*.exe"
mkdir obj

echo Detectando GPU CUDA...
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>nul
if %ERRORLEVEL% neq 0 (
    echo ⚠️  nvidia-smi no disponible, pero continuando con CUDA...
)

echo.
echo Compilando operaciones CUDA optimizadas...
REM Usar CUDA con compute capability compatible
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math --ptxas-options=-v ^
     -gencode arch=compute_61,code=sm_61 ^
     -gencode arch=compute_75,code=sm_75 ^
     -Xcompiler "/MD" -Iinclude ^
     -c src/matrix_cuda.cu -o obj/matrix_cuda.o

if %ERRORLEVEL% neq 0 (
    echo ❌ NVCC falló. Usando versión CPU optimizada con OpenMP...
    echo Compilando versión CPU con paralelización...
    g++ -std=c++17 -O3 -fopenmp -DUSE_CUDA -DCPU_PARALLEL -Iinclude ^
        -c src/matrix_cuda_fallback.cpp -o obj/matrix_cuda.o
    if %ERRORLEVEL% neq 0 (
        echo ❌ Error en compilación fallback
        pause
        exit /b 1
    )
    set CUDA_MODE=CPU_PARALLEL
) else (
    echo ✅ CUDA compilado exitosamente
    set CUDA_MODE=CUDA_GPU
)

echo.
echo Compilando archivos C++ del Transformer...
g++ -std=c++17 -O3 -fopenmp -DUSE_CUDA -Iinclude -c src/matrix.cpp -o obj/matrix.o
g++ -std=c++17 -O3 -fopenmp -DUSE_CUDA -Iinclude -c src/mnist_loader.cpp -o obj/mnist_loader.o  
g++ -std=c++17 -O3 -fopenmp -DUSE_CUDA -Iinclude -c src/transformer.cpp -o obj/transformer.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando archivos C++
    pause
    exit /b 1
)

echo ✅ Archivos C++ compilados

echo.
echo Enlazando Transformer optimizado...
if "%CUDA_MODE%"=="CUDA_GPU" (
    echo Enlazando con librerías CUDA...
    g++ -std=c++17 -O3 -fopenmp -DUSE_CUDA -Iinclude ^
        -o TransformerCUDA_Fast.exe main_fast.cpp ^
        obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o ^
        -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64" ^
        -lcudart -lcublas -lcurand
        
    if %ERRORLEVEL% neq 0 (
        echo Reintentando enlazado simple...
        g++ -std=c++17 -O3 -fopenmp -DUSE_CUDA -Iinclude ^
            -o TransformerCUDA_Fast.exe main_fast.cpp ^
            obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o ^
            -lcudart
    )
) else (
    echo Enlazando versión CPU paralelizada...
    g++ -std=c++17 -O3 -fopenmp -DUSE_CUDA -Iinclude ^
        -o TransformerCUDA_Fast.exe main_fast.cpp ^
        obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o
)

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado
    pause
    exit /b 1
)

echo.
echo 🚀 ¡Transformer CUDA compilado exitosamente!
echo.
echo Modo de aceleración: %CUDA_MODE%
echo Ejecutable: TransformerCUDA_Fast.exe
echo.
echo 📊 Optimizaciones incluidas:
echo   - Operaciones matriciales CUDA/paralelas
echo   - Batch processing optimizado (tamaño 256)
echo   - Reducción de épocas para pruebas rápidas
echo   - Memory pooling para GPU
echo.
echo Para ejecutar:
echo   .\TransformerCUDA_Fast.exe
echo.
pause
