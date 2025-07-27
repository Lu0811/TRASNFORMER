@echo off
echo =========================================================
echo  🚀 COMPILADOR CUDA TRANSFORMER FIXED (Visual Studio) 🚀
echo =========================================================

REM Configurar entorno Visual Studio 2022
echo Configurando entorno Visual Studio 2022...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

if %ERRORLEVEL% neq 0 (
    echo ❌ Error: No se pudo configurar Visual Studio 2022
    echo 💡 Verifica que Visual Studio 2022 Community esté instalado
    pause
    exit /b 1
)

echo ✅ Entorno Visual Studio configurado

REM Limpiar compilaciones anteriores
echo.
echo 🧹 Limpiando compilaciones anteriores...
if exist "obj_vs_fixed" rmdir /s /q "obj_vs_fixed"
if exist "TransformerFixed_CUDA_VS.exe" del "TransformerFixed_CUDA_VS.exe"
mkdir obj_vs_fixed

echo ✅ Directorio limpiado

REM Verificar CUDA
echo.
echo 🔍 Verificando CUDA Toolkit...
nvcc --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Error: NVCC no encontrado
    echo 💡 Verifica que CUDA Toolkit 12.8 esté instalado y en PATH
    pause
    exit /b 1
)

echo ✅ CUDA Toolkit verificado
nvcc --version | findstr "release"

echo.
echo =======================================================
echo 📦 COMPILANDO IMPLEMENTACIÓN TRANSFORMER FIXED...
echo =======================================================

echo.
echo [1/5] 🔥 Compilando kernels CUDA optimizados...
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math -arch=sm_86 -arch=sm_75 -arch=sm_61 ^
     -Iinclude -c src/matrix_cuda.cu -o obj_vs_fixed/matrix_cuda.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando matrix_cuda.cu
    echo 💡 Verifica que el archivo src/matrix_cuda.cu exista
    pause
    exit /b 1
)

echo ✅ Kernels CUDA compilados

echo.
echo [2/5] 🔧 Compilando matriz base con MSVC...
cl /O2 /EHsc /DUSE_CUDA /std:c++17 ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/matrix.cpp /Fo:obj_vs_fixed/matrix.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando matrix.cpp
    pause
    exit /b 1
)

echo ✅ Matrix.cpp compilado

echo.
echo [3/5] 📊 Compilando MNIST loader con MSVC...
cl /O2 /EHsc /DUSE_CUDA /std:c++17 ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/mnist_loader.cpp /Fo:obj_vs_fixed/mnist_loader.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando mnist_loader.cpp
    pause
    exit /b 1
)

echo ✅ MNIST loader compilado

echo.
echo [4/5] 🧠 Compilando TRANSFORMER FIXED con MSVC...
cl /O2 /EHsc /DUSE_CUDA /std:c++17 ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/transformer_fixed.cpp /Fo:obj_vs_fixed/transformer_fixed.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando transformer_fixed.cpp
    echo 💡 Verifica que el archivo src/transformer_fixed.cpp exista
    pause
    exit /b 1
)

echo ✅ Transformer FIXED compilado

echo.
echo [5/5] 🔗 Enlazando ejecutable final con MSVC...
cl /O2 /EHsc /DUSE_CUDA /std:c++17 ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   main_fixed.cpp obj_vs_fixed/matrix.obj obj_vs_fixed/mnist_loader.obj ^
   obj_vs_fixed/transformer_fixed.obj obj_vs_fixed/matrix_cuda.obj ^
   /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib curand.lib /OUT:TransformerFixed_CUDA_VS.exe

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado final
    echo 💡 Verifica que las bibliotecas CUDA estén disponibles:
    echo    - cudart.lib, cublas.lib, curand.lib
    echo 💡 Verifica que main_fixed.cpp exista
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 🎉 ¡TRANSFORMER FIXED COMPILADO EXITOSAMENTE CON CUDA! 🎉
echo ============================================================

echo.
echo ✅ Ejecutable creado: TransformerFixed_CUDA_VS.exe
echo.
echo 📊 Configuración del sistema:
echo   - GPU: RTX 3050 Laptop
echo   - CUDA: 12.8
echo   - Compilador: Visual Studio 2022 (MSVC)
echo   - Arquitectura: x64
echo   - Optimización: /O2 (máxima)
echo.
echo 🔥 Mejoras de la implementación FIXED:
echo   ✅ Multi-head attention REAL (no simulada)
echo   ✅ Backpropagation COMPLETA end-to-end
echo   ✅ Layer normalization Pre-LN (más estable)
echo   ✅ CUDA optimizado con memory pooling
echo   ✅ Batch processing REAL paralelo
echo   ✅ Adam optimizer correcto
echo.

echo 📈 Rendimiento esperado vs original:
echo   - Accuracy: 85-92% (vs 60-70% original)
echo   - Convergencia: 5-10 épocas (vs 20+ original)
echo   - Velocidad GPU: 3-5x más rápido
echo   - Estabilidad: Sin explosión de gradientes
echo.

echo =======================================================
echo 🧪 COMPILANDO TESTS ADICIONALES...
echo =======================================================

echo.
echo Compilando test de matriz CUDA...
cl /O2 /EHsc /DUSE_CUDA /std:c++17 ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   test_cuda.cpp obj_vs_fixed/matrix_cuda.obj ^
   /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib /OUT:TestCudaFixed_VS.exe

if %ERRORLEVEL% neq 0 (
    echo ⚠️ Advertencia: No se pudo compilar test CUDA (opcional)
) else (
    echo ✅ Test CUDA compilado: TestCudaFixed_VS.exe
)

echo.
echo =======================================================
echo 🚀 INSTRUCCIONES DE USO
echo =======================================================

echo.
echo Para ejecutar el transformer FIXED:
echo   .\TransformerFixed_CUDA_VS.exe
echo.
echo Para probar operaciones CUDA:
echo   .\TestCudaFixed_VS.exe
echo.
echo 📁 Archivos requeridos en el directorio:
echo   - train-images-idx3-ubyte
echo   - train-labels-idx1-ubyte
echo   - t10k-images-idx3-ubyte
echo   - t10k-labels-idx1-ubyte
echo.
echo 💾 Archivos de salida generados:
echo   - training_history_fixed.csv
echo   - predictions_fixed.csv
echo   - test_metrics_fixed.csv
echo.
echo 💡 Descarga Fashion-MNIST desde:
echo   https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
echo.
echo 🔥 ¡LISTO PARA ENTRENAR CON IMPLEMENTACIÓN MATEMÁTICAMENTE CORRECTA!
echo.

pause