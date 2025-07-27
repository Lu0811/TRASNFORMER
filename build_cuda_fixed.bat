@echo off
echo ===================================================
echo  🚀 BUILD TRANSFORMER CORREGIDO CON CUDA v3.0 🚀
echo ===================================================

REM Verificar que NVCC esté disponible
echo Verificando CUDA...
nvcc --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Error: NVCC no encontrado. Asegúrate de tener CUDA instalado y en el PATH.
    echo 💡 Descarga CUDA desde: https://developer.nvidia.com/cuda-toolkit
    pause
    exit /b 1
)

echo ✅ CUDA Toolkit encontrado
nvcc --version | findstr "release"

REM Verificar que g++ esté disponible
echo.
echo Verificando compilador C++...
g++ --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Error: g++ no encontrado. Instala MinGW-w64 o Visual Studio Build Tools.
    pause
    exit /b 1
)

echo ✅ Compilador C++ encontrado

REM Crear directorio obj si no existe
if not exist "obj" mkdir obj

echo.
echo ==============================================
echo 📦 COMPILANDO IMPLEMENTACIÓN CORREGIDA...
echo ==============================================

echo.
echo [1/5] Compilando operaciones CUDA optimizadas...
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math -arch=sm_86 -arch=sm_75 -arch=sm_61 -Iinclude ^
     -c src/matrix_cuda.cu -o obj/matrix_cuda.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando matrix_cuda.cu
    pause
    exit /b 1
)
echo ✅ CUDA operations compiled

echo.
echo [2/5] Compilando matriz base...
g++ -std=c++17 -O3 -Wall -Wextra -DUSE_CUDA -Iinclude ^
    -c src/matrix.cpp -o obj/matrix.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando matrix.cpp
    pause
    exit /b 1
)
echo ✅ Matrix base compiled

echo.
echo [3/5] Compilando MNIST loader...
g++ -std=c++17 -O3 -Wall -Wextra -DUSE_CUDA -Iinclude ^
    -c src/mnist_loader.cpp -o obj/mnist_loader.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando mnist_loader.cpp
    pause
    exit /b 1
)
echo ✅ MNIST loader compiled

echo.
echo [4/5] Compilando TRANSFORMER CORREGIDO...
g++ -std=c++17 -O3 -Wall -Wextra -DUSE_CUDA -Iinclude ^
    -c src/transformer_fixed.cpp -o obj/transformer_fixed.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando transformer_fixed.cpp
    pause
    exit /b 1
)
echo ✅ Fixed Transformer compiled

echo.
echo [5/5] Enlazando ejecutable final...
g++ -std=c++17 -O3 -Wall -Wextra -DUSE_CUDA -Iinclude ^
    -o TransformerFixed_CUDA.exe main.cpp ^
    obj/matrix.o obj/mnist_loader.o obj/transformer_fixed.o obj/matrix_cuda.o ^
    -lcudart -lcublas

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado final
    echo 💡 Verifica que CUDA libs estén en PATH: cudart64_12.dll, cublas64_12.dll
    pause
    exit /b 1
)

echo.
echo ================================================
echo 🎉 ¡COMPILACIÓN EXITOSA! 🎉
echo ================================================

echo.
echo ✅ Ejecutable creado: TransformerFixed_CUDA.exe
echo 📊 Tamaño del ejecutable:
dir TransformerFixed_CUDA.exe | findstr "TransformerFixed_CUDA.exe"

echo.
echo ==============================================
echo 🧪 COMPILANDO TESTS OPCIONALES...
echo ==============================================

echo.
echo Compilando test de matrices CUDA...
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude ^
    -o test_cuda_matrix.exe test_cuda.cpp obj/matrix_cuda.o -lcudart

if %ERRORLEVEL% neq 0 (
    echo ⚠️ Advertencia: No se pudo compilar test_cuda_matrix.exe
) else (
    echo ✅ Test CUDA compilado: test_cuda_matrix.exe
)

echo.
echo ==============================================
echo 🚀 INSTRUCCIONES DE USO
echo ==============================================

echo.
echo Para ejecutar el transformer corregido:
echo   .\TransformerFixed_CUDA.exe
echo.
echo Para probar operaciones CUDA:
echo   .\test_cuda_matrix.exe
echo.
echo 📁 Archivos requeridos en el directorio:
echo   - train-images-idx3-ubyte
echo   - train-labels-idx1-ubyte  
echo   - t10k-images-idx3-ubyte
echo   - t10k-labels-idx1-ubyte
echo.
echo 💡 Descarga Fashion-MNIST desde:
echo   https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
echo.

pause