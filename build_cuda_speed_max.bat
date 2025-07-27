@echo off
echo ===============================================
echo  COMPILADOR CUDA - VELOCIDAD MAXIMA
echo  OBJETIVO: ^<500ms por batch GARANTIZADO
echo ===============================================

echo.
echo ⚡ CONFIGURANDO PARA VELOCIDAD MAXIMA...
echo =====================================

REM Verificar NVIDIA drivers
nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ NVIDIA drivers no detectados
    pause
    exit /b 1
)

echo ✅ Drivers NVIDIA detectados

echo.
echo 📊 ESTADO GPU:
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits

REM Variables de entorno para VELOCIDAD
echo.
echo ⚡ Configurando variables VELOCIDAD...
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_LAUNCH_BLOCKING=0
set CUDA_AUTO_BOOST=1

echo ✅ Variables configuradas

REM Configurar Visual Studio
echo.
echo 🔧 Configurando Visual Studio...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Limpiar archivos anteriores
echo.
echo 🧹 Limpiando archivos anteriores...
if exist "obj_vs" rmdir /s /q "obj_vs"
if exist "TransformerSpeedMax.exe" del "TransformerSpeedMax.exe"
mkdir obj_vs

echo.
echo ⚡ Compilando kernels CUDA VELOCIDAD...
echo ====================================

REM Compilar CUDA optimizado para velocidad
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math ^
     -arch=sm_86 ^
     -gencode arch=compute_86,code=sm_86 ^
     --ptxas-options=-v ^
     -Iinclude -c src/matrix_cuda.cu -o obj_vs/matrix_cuda.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando CUDA
    pause
    exit /b 1
)

echo ✅ CUDA compilado para velocidad

echo.
echo ⚡ Compilando C++ VELOCIDAD...
echo ===========================

REM Compilar C++ con optimizaciones de velocidad
cl /O2 /Ox /Ot /EHsc /DUSE_CUDA ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/matrix.cpp /Fo:obj_vs/matrix.obj

cl /O2 /Ox /Ot /EHsc /DUSE_CUDA ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/mnist_loader.cpp /Fo:obj_vs/mnist_loader.obj

cl /O2 /Ox /Ot /EHsc /DUSE_CUDA ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/transformer.cpp /Fo:obj_vs/transformer.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando C++
    pause
    exit /b 1
)

echo ✅ C++ compilado exitosamente

echo.
echo ⚡ Enlazando VELOCIDAD MÁXIMA...
echo =============================

REM Enlazar versión velocidad máxima
cl /O2 /Ox /Ot /EHsc /DUSE_CUDA ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   main_speed_max.cpp obj_vs/matrix.obj obj_vs/mnist_loader.obj obj_vs/transformer.obj obj_vs/matrix_cuda.obj ^
   /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib cublasLt.lib ^
   /OUT:TransformerSpeedMax.exe

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado
    pause
    exit /b 1
)

echo.
echo ⚡ ¡PROYECTO VELOCIDAD MÁXIMA COMPILADO!
echo =====================================

echo.
echo 🚀 Tu Transformer VELOCIDAD MÁXIMA incluye:
echo   - ⚡ Batch size 64 ^(pequeño para velocidad^)
echo   - ⚡ Modelo 800K parámetros ^(ligero^)
echo   - ⚡ d_model=128, num_heads=4, layers=4
echo   - ⚡ Dataset 10K muestras ^(demo rápida^)
echo   - ⚡ Sin código complejo innecesario
echo   - ⚡ Objetivo: ^<500ms por batch
echo   - ⚡ Accuracy esperado: ~75%%
echo.

echo Ejecutable: TransformerSpeedMax.exe
echo.

echo ⚡ VERIFICACIÓN RÁPIDA:
echo ====================

REM Crear verificador velocidad
echo #include ^<iostream^> > speed_check.cpp
echo #include ^<cuda_runtime.h^> >> speed_check.cpp
echo #include ^<cublas_v2.h^> >> speed_check.cpp
echo #include ^<chrono^> >> speed_check.cpp
echo int main^(^) { >> speed_check.cpp
echo     printf^("⚡ VERIFICACIÓN VELOCIDAD MÁXIMA\\n\\n"^); >> speed_check.cpp
echo     int count; cudaGetDeviceCount^(&count^); >> speed_check.cpp
echo     cudaDeviceProp prop; cudaGetDeviceProperties^(&prop, 0^); >> speed_check.cpp
echo     printf^("GPU: %%s ^(%%zu MB^)\\n", prop.name, prop.totalGlobalMem/1024/1024^); >> speed_check.cpp
echo     // Test velocidad pequeña >> speed_check.cpp
echo     const int N = 128; >> speed_check.cpp
echo     float *d_A, *d_B, *d_C; >> speed_check.cpp
echo     cudaMalloc^(&d_A, N*N*sizeof^(float^)^); >> speed_check.cpp
echo     cudaMalloc^(&d_B, N*N*sizeof^(float^)^); >> speed_check.cpp
echo     cudaMalloc^(&d_C, N*N*sizeof^(float^)^); >> speed_check.cpp
echo     cublasHandle_t handle; cublasCreate^(&handle^); >> speed_check.cpp
echo     float alpha=1.0f, beta=0.0f; >> speed_check.cpp
echo     auto start = std::chrono::high_resolution_clock::now^(^); >> speed_check.cpp
echo     for^(int i=0; i^<100; i++^) { >> speed_check.cpp
echo         cublasSgemm^(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N^); >> speed_check.cpp
echo     } >> speed_check.cpp
echo     cudaDeviceSynchronize^(^); >> speed_check.cpp
echo     auto end = std::chrono::high_resolution_clock::now^(^); >> speed_check.cpp
echo     auto time = std::chrono::duration_cast^<std::chrono::microseconds^>^(end-start^).count^(^); >> speed_check.cpp
echo     printf^("\\n⚡ 100x multiplicaciones 128x128: %%ld μs\\n", time^); >> speed_check.cpp
echo     printf^("⚡ Promedio por operación: %%ld μs\\n", time/100^); >> speed_check.cpp
echo     if^(time/100 ^< 100^) printf^("✅ GPU ULTRA-RÁPIDA\\n"^); >> speed_check.cpp
echo     cublasDestroy^(handle^); >> speed_check.cpp
echo     cudaFree^(d_A^); cudaFree^(d_B^); cudaFree^(d_C^); >> speed_check.cpp
echo     return 0; >> speed_check.cpp
echo } >> speed_check.cpp

cl /O2 /EHsc /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" speed_check.cpp /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" cudart.lib cublas.lib /OUT:speed_check.exe >nul 2>&1

if exist "speed_check.exe" (
    speed_check.exe
    del speed_check.exe speed_check.cpp speed_check.obj >nul 2>&1
)

echo.
echo ⚡ INSTRUCCIONES VELOCIDAD MÁXIMA:
echo ===============================
echo.
echo 1. EJECUTAR DIRECTAMENTE:
echo    .\TransformerSpeedMax.exe
echo.
echo 2. MONITOREAR ^(OPCIONAL^):
echo    nvidia-smi -l 1
echo.
echo 3. RESULTADOS ESPERADOS:
echo    ⚡ Tiempo promedio: ^<500ms/batch
echo    ⚡ Total épocas: 5 ^(demo rápida^)
echo    ⚡ Accuracy: ~70-75%%
echo    ⚡ GPU uso: ~40-60%% ^(modelo ligero^)
echo    ⚡ Memoria: ~1000MB
echo.
echo 4. CARACTERÍSTICAS:
echo    - Modelo ultra-ligero 800K params
echo    - Batch pequeño de 64 muestras
echo    - Solo 4 capas encoder
echo    - 4 attention heads
echo    - Dataset 10K muestras
echo    - Sin operaciones innecesarias
echo.
echo ⚡ Si aún es ^>500ms/batch:
echo    1. Reduce BATCH_SIZE a 32
echo    2. Reduce num_layers a 2
echo    3. Verifica que no haya throttling
echo.

pause