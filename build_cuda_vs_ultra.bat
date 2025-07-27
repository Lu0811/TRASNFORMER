@echo off
echo ===============================================
echo  COMPILADOR CUDA ULTRA-OPTIMIZADO
echo  VELOCIDAD MAXIMA ^<1000ms/batch + GPU 80%+
echo ===============================================

echo.
echo ⚡ CONFIGURANDO PARA VELOCIDAD ULTRA + SATURACION GPU...
echo =========================================================

REM Verificar NVIDIA drivers
nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ NVIDIA drivers no detectados
    pause
    exit /b 1
)

echo ✅ Drivers NVIDIA detectados

echo.
echo 📊 ESTADO GPU INICIAL:
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits

REM Variables de entorno CRÍTICAS para VELOCIDAD MÁXIMA + SATURACIÓN
echo.
echo ⚡ Configurando variables ULTRA-VELOCIDAD...
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set NVIDIA_VISIBLE_DEVICES=all
set NVIDIA_DRIVER_CAPABILITIES=compute,utility

REM Variables específicas para VELOCIDAD MÁXIMA
set CUDA_LAUNCH_BLOCKING=0
set CUDA_CACHE_DISABLE=0
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_WAITS_ON_EXCEPTION=0

REM Variables para SATURACIÓN GPU
set GPU_FORCE_64BIT_PTR=0
set GPU_MAX_HEAP_SIZE=100
set GPU_MAX_ALLOC_PERCENT=90
set GPU_SINGLE_ALLOC_PERCENT=90

REM Variables para MEMORIA OPTIMIZADA
set CUDA_DEVICE_MAX_CONNECTIONS=32
set CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT=90

echo ✅ Variables ULTRA-VELOCIDAD configuradas

REM Configurar Visual Studio
echo.
echo 🔧 Configurando Visual Studio...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Limpiar archivos anteriores
echo.
echo 🧹 Limpiando archivos anteriores...
if exist "obj_vs" rmdir /s /q "obj_vs"
if exist "TransformerUltra.exe" del "TransformerUltra.exe"
mkdir obj_vs

echo.
echo ⚡ Compilando kernels CUDA ULTRA-OPTIMIZADOS...
echo ==============================================

REM Compilar CUDA con FLAGS DE VELOCIDAD MÁXIMA
nvcc -std=c++17 -O3 -DUSE_CUDA -DFORCE_NVIDIA_GPU -DGPU_ULTRA_MODE --use_fast_math ^
     -arch=sm_86 ^
     -gencode arch=compute_86,code=sm_86 ^
     --ptxas-options=-v,--opt-level=3,--allow-expensive-optimizations=true ^
     --gpu-architecture=sm_86 ^
     --maxrregcount=32 ^
     --ftz=true ^
     --prec-div=false ^
     --prec-sqrt=false ^
     --fmad=true ^
     --extra-device-vectorization ^
     --restrict ^
     --relocatable-device-code=false ^
     -Iinclude -c src/matrix_cuda.cu -o obj_vs/matrix_cuda.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando CUDA
    pause
    exit /b 1
)

echo ✅ CUDA compilado con ULTRA-OPTIMIZACIONES

echo.
echo ⚡ Compilando C++ con VELOCIDAD MÁXIMA...
echo ========================================

REM Compilar con TODAS las optimizaciones de velocidad posibles
cl /O2 /Ox /Ot /Ob2 /Oi /Oy /GL /GS- /Gy /GF /GA /favor:INTEL64 /arch:AVX2 ^
   /fp:fast /fp:except- /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /DGPU_ULTRA_MODE ^
   /DNDEBUG /DWIN32_LEAN_AND_MEAN /DNOMINMAX ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/matrix.cpp /Fo:obj_vs/matrix.obj

cl /O2 /Ox /Ot /Ob2 /Oi /Oy /GL /GS- /Gy /GF /GA /favor:INTEL64 /arch:AVX2 ^
   /fp:fast /fp:except- /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /DGPU_ULTRA_MODE ^
   /DNDEBUG /DWIN32_LEAN_AND_MEAN /DNOMINMAX ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/mnist_loader.cpp /Fo:obj_vs/mnist_loader.obj

cl /O2 /Ox /Ot /Ob2 /Oi /Oy /GL /GS- /Gy /GF /GA /favor:INTEL64 /arch:AVX2 ^
   /fp:fast /fp:except- /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /DGPU_ULTRA_MODE ^
   /DNDEBUG /DWIN32_LEAN_AND_MEAN /DNOMINMAX ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/transformer.cpp /Fo:obj_vs/transformer.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando C++
    pause
    exit /b 1
)

echo ✅ C++ compilado con VELOCIDAD MÁXIMA

echo.
echo ⚡ Enlazando con TODAS las librerías ULTRA...
echo ============================================

REM Enlazar con TODAS las optimizaciones y librerías disponibles
cl /O2 /Ox /Ot /Ob2 /Oi /Oy /GL /GS- /Gy /GF /GA /favor:INTEL64 /arch:AVX2 ^
   /fp:fast /fp:except- /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /DGPU_ULTRA_MODE ^
   /DNDEBUG /DWIN32_LEAN_AND_MEAN /DNOMINMAX ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   main_ultra_optimized.cpp obj_vs/matrix.obj obj_vs/mnist_loader.obj obj_vs/transformer.obj obj_vs/matrix_cuda.obj ^
   /link /LTCG /OPT:REF /OPT:ICF /INCREMENTAL:NO /SUBSYSTEM:CONSOLE ^
   /MACHINE:X64 /LARGEADDRESSAWARE ^
   /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib cublasLt.lib cusparse.lib curand.lib cusolver.lib cufft.lib ^
   kernel32.lib user32.lib advapi32.lib ^
   /OUT:TransformerUltra.exe

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado
    pause
    exit /b 1
)

echo.
echo ⚡ ¡PROYECTO ULTRA-OPTIMIZADO COMPILADO!
echo =======================================

echo.
echo 🚀 Tu Transformer ULTRA incluye:
echo   - ⚡ Batch size 256 ^(4x más grande que original^)
echo   - ⚡ Modelo 6M parámetros ^(para saturar GPU completa^)
echo   - ⚡ d_model=384, num_heads=12, layers=8
echo   - ⚡ Memory pooling GPU ^(sin re-allocaciones^)
echo   - ⚡ Procesamiento paralelo de batches
echo   - ⚡ Streams asíncronos CUDA
echo   - ⚡ Normalización paralela CPU
echo   - ⚡ Cache L1 optimizado
echo   - ⚡ Todas las optimizaciones MSVC/NVCC
echo   - ⚡ Variables entorno máximo rendimiento
echo.

echo Ejecutable: TransformerUltra.exe
echo.

echo ⚡ VERIFICACIÓN ULTRA-RÁPIDA:
echo ============================

REM Crear verificador de configuración ultra
echo #include ^<iostream^> > ultra_check.cpp
echo #include ^<cuda_runtime.h^> >> ultra_check.cpp
echo #include ^<cublas_v2.h^> >> ultra_check.cpp
echo #include ^<chrono^> >> ultra_check.cpp
echo int main^(^) { >> ultra_check.cpp
echo     printf^("⚡ VERIFICACIÓN ULTRA-OPTIMIZADA\\n"^); >> ultra_check.cpp
echo     int count; cudaGetDeviceCount^(&count^); >> ultra_check.cpp
echo     for^(int i=0; i^<count; i++^) { >> ultra_check.cpp
echo         cudaDeviceProp prop; cudaGetDeviceProperties^(&prop, i^); >> ultra_check.cpp
echo         printf^("GPU %%d: %%s\\n", i, prop.name^); >> ultra_check.cpp
echo         printf^("  Max threads/block: %%d\\n", prop.maxThreadsPerBlock^); >> ultra_check.cpp
echo         printf^("  Shared mem/block: %%zu KB\\n", prop.sharedMemPerBlock/1024^); >> ultra_check.cpp
echo         printf^("  Registers/block: %%d\\n", prop.regsPerBlock^); >> ultra_check.cpp
echo         if^(prop.totalGlobalMem ^> 3000000000^) { >> ultra_check.cpp
echo             printf^("  ⚡ GPU DEDICADA - ULTRA-OPTIMIZADA\\n"^); >> ultra_check.cpp
echo         } >> ultra_check.cpp
echo     } >> ultra_check.cpp
echo     // Test velocidad cuBLAS >> ultra_check.cpp
echo     const int N = 1024; >> ultra_check.cpp
echo     float *d_A, *d_B, *d_C; >> ultra_check.cpp
echo     cudaMalloc^(&d_A, N*N*sizeof^(float^)^); >> ultra_check.cpp
echo     cudaMalloc^(&d_B, N*N*sizeof^(float^)^); >> ultra_check.cpp
echo     cudaMalloc^(&d_C, N*N*sizeof^(float^)^); >> ultra_check.cpp
echo     cublasHandle_t handle; cublasCreate^(&handle^); >> ultra_check.cpp
echo     float alpha=1.0f, beta=0.0f; >> ultra_check.cpp
echo     auto start = std::chrono::high_resolution_clock::now^(^); >> ultra_check.cpp
echo     cublasSgemm^(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N^); >> ultra_check.cpp
echo     cudaDeviceSynchronize^(^); >> ultra_check.cpp
echo     auto end = std::chrono::high_resolution_clock::now^(^); >> ultra_check.cpp
echo     auto time = std::chrono::duration_cast^<std::chrono::microseconds^>^(end-start^).count^(^); >> ultra_check.cpp
echo     printf^("\\n⚡ cuBLAS 1024x1024: %%ld μs\\n", time^); >> ultra_check.cpp
echo     if^(time ^< 5000^) printf^("✅ GPU ULTRA-RÁPIDA\\n"^); >> ultra_check.cpp
echo     else printf^("⚠️  GPU velocidad normal\\n"^); >> ultra_check.cpp
echo     cublasDestroy^(handle^); >> ultra_check.cpp
echo     cudaFree^(d_A^); cudaFree^(d_B^); cudaFree^(d_C^); >> ultra_check.cpp
echo     return 0; >> ultra_check.cpp
echo } >> ultra_check.cpp

cl /O2 /EHsc /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ultra_check.cpp /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" cudart.lib cublas.lib /OUT:ultra_check.exe >nul 2>&1

if exist "ultra_check.exe" (
    ultra_check.exe
    del ultra_check.exe ultra_check.cpp ultra_check.obj >nul 2>&1
)

echo.
echo ⚡ INSTRUCCIONES ULTRA-OPTIMIZADAS:
echo ==================================
echo.
echo 1. ABRIR MONITOREO ^(OBLIGATORIO^):
echo    nvidia-smi -l 1
echo.
echo 2. EJECUTAR ULTRA-OPTIMIZADO:
echo    .\TransformerUltra.exe
echo.
echo 3. RESULTADOS ESPERADOS:
echo    ⚡ Tiempo por batch: ^<1000ms ^(vs 8000ms anterior^)
echo    ⚡ GPU Utilization: ^>80%% ^(vs 20%% anterior^)
echo    ⚡ Memory Usage: ^>3000MB ^(vs 800MB anterior^)
echo    ⚡ Power Usage: ^>50W ^(vs 7W anterior^)
echo    ⚡ Performance: P0 máximo
echo.
echo 4. OPTIMIZACIONES APLICADAS:
echo    - Batch size 256 ^(4x más grande^)
echo    - Modelo 6M params ^(8x más grande^)
echo    - Memory pooling GPU
echo    - Procesamiento paralelo
echo    - Streams asíncronos
echo    - Todas las optimizaciones compiler
echo.
echo ⚠️  Si el batch sigue ^>2000ms:
echo 1. Reduce BATCH_SIZE a 128 en el código
echo 2. Recompila con este script
echo 3. Verifica thermal throttling en GPU
echo.

pause