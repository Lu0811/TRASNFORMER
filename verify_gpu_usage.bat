@echo off
echo ================================================================
echo  VERIFICADOR DE USO DE GPU NVIDIA DEDICADA
echo ================================================================

echo.
echo 🔍 1. DETECTANDO GPUS DISPONIBLES...
echo ----------------------------------------
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory --format=csv,noheader,nounits

echo.
echo 🔍 2. VERIFICANDO CONFIGURACIÓN CUDA...
echo ----------------------------------------
nvcc --version
echo.
nvidia-smi

echo.
echo 🔍 3. COMPILANDO VERIFICADOR DE GPU...
echo ----------------------------------------

REM Configurar Visual Studio
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Limpiar archivos anteriores
if exist "gpu_monitor.exe" del "gpu_monitor.exe"
if exist "gpu_test.obj" del "gpu_test.obj"

echo.
echo ✅ Compilando verificador GPU dedicada...

REM Crear código C++ para verificar GPU
echo #include ^<iostream^> > gpu_monitor.cpp
echo #include ^<cuda_runtime.h^> >> gpu_monitor.cpp
echo #include ^<cublas_v2.h^> >> gpu_monitor.cpp
echo #include ^<chrono^> >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo void force_nvidia_gpu() { >> gpu_monitor.cpp
echo     int deviceCount = 0; >> gpu_monitor.cpp
echo     cudaGetDeviceCount^(&deviceCount^); >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     std::cout ^<^< "\\n🔍 DETECTADAS " ^<^< deviceCount ^<^< " GPUS CUDA:\\n"; >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     for ^(int i = 0; i ^< deviceCount; i++^) { >> gpu_monitor.cpp
echo         cudaDeviceProp prop; >> gpu_monitor.cpp
echo         cudaGetDeviceProperties^(&prop, i^); >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo         std::cout ^<^< "GPU " ^<^< i ^<^< ": " ^<^< prop.name ^<^< std::endl; >> gpu_monitor.cpp
echo         std::cout ^<^< "   Memoria total: " ^<^< prop.totalGlobalMem / ^(1024*1024^) ^<^< " MB" ^<^< std::endl; >> gpu_monitor.cpp
echo         std::cout ^<^< "   Compute Capability: " ^<^< prop.major ^<^< "." ^<^< prop.minor ^<^< std::endl; >> gpu_monitor.cpp
echo         std::cout ^<^< "   Multiprocessors: " ^<^< prop.multiProcessorCount ^<^< std::endl; >> gpu_monitor.cpp
echo         std::cout ^<^< "   Clock Rate: " ^<^< prop.clockRate / 1000 ^<^< " MHz" ^<^< std::endl; >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo         // Detectar si es GPU dedicada ^(más memoria^) >> gpu_monitor.cpp
echo         if ^(prop.totalGlobalMem ^> 2000000000^) { >> gpu_monitor.cpp
echo             std::cout ^<^< "   ⭐ GPU DEDICADA DETECTADA" ^<^< std::endl; >> gpu_monitor.cpp
echo         } else { >> gpu_monitor.cpp
echo             std::cout ^<^< "   ⚠️  Posible GPU integrada" ^<^< std::endl; >> gpu_monitor.cpp
echo         } >> gpu_monitor.cpp
echo         std::cout ^<^< std::endl; >> gpu_monitor.cpp
echo     } >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     // FORZAR USO DE GPU MÁS POTENTE >> gpu_monitor.cpp
echo     int bestGPU = 0; >> gpu_monitor.cpp
echo     size_t maxMemory = 0; >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     for ^(int i = 0; i ^< deviceCount; i++^) { >> gpu_monitor.cpp
echo         cudaDeviceProp prop; >> gpu_monitor.cpp
echo         cudaGetDeviceProperties^(&prop, i^); >> gpu_monitor.cpp
echo         if ^(prop.totalGlobalMem ^> maxMemory^) { >> gpu_monitor.cpp
echo             maxMemory = prop.totalGlobalMem; >> gpu_monitor.cpp
echo             bestGPU = i; >> gpu_monitor.cpp
echo         } >> gpu_monitor.cpp
echo     } >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     std::cout ^<^< "🎯 SELECCIONANDO GPU " ^<^< bestGPU ^<^< " ^(más memoria: " ^<^< maxMemory/^(1024*1024^) ^<^< "MB^)\\n"; >> gpu_monitor.cpp
echo     cudaSetDevice^(bestGPU^); >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     cudaDeviceProp activeProp; >> gpu_monitor.cpp
echo     cudaGetDeviceProperties^(&activeProp, bestGPU^); >> gpu_monitor.cpp
echo     std::cout ^<^< "✅ GPU ACTIVA: " ^<^< activeProp.name ^<^< "\\n\\n"; >> gpu_monitor.cpp
echo } >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo void stress_test_gpu() { >> gpu_monitor.cpp
echo     std::cout ^<^< "🔥 EJECUTANDO STRESS TEST GPU...\\n"; >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     // Allocate large matrices on GPU >> gpu_monitor.cpp
echo     const int size = 2048; >> gpu_monitor.cpp
echo     const int elements = size * size; >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     float *d_A, *d_B, *d_C; >> gpu_monitor.cpp
echo     cudaMalloc^(&d_A, elements * sizeof^(float^)^); >> gpu_monitor.cpp
echo     cudaMalloc^(&d_B, elements * sizeof^(float^)^); >> gpu_monitor.cpp
echo     cudaMalloc^(&d_C, elements * sizeof^(float^)^); >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     // Initialize with cuBLAS >> gpu_monitor.cpp
echo     cublasHandle_t handle; >> gpu_monitor.cpp
echo     cublasCreate^(&handle^); >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     // Fill matrices >> gpu_monitor.cpp
echo     float alpha = 1.0f, beta = 0.0f; >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     auto start = std::chrono::high_resolution_clock::now^(^); >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     // Multiple heavy operations to stress GPU >> gpu_monitor.cpp
echo     for ^(int i = 0; i ^< 10; i++^) { >> gpu_monitor.cpp
echo         cublasSgemm^(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, d_A, size, d_B, size, &beta, d_C, size^); >> gpu_monitor.cpp
echo         cudaDeviceSynchronize^(^); >> gpu_monitor.cpp
echo     } >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     auto end = std::chrono::high_resolution_clock::now^(^); >> gpu_monitor.cpp
echo     auto duration = std::chrono::duration_cast^<std::chrono::milliseconds^>^(end - start^).count^(^); >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     std::cout ^<^< "⏱️  Tiempo stress test: " ^<^< duration ^<^< " ms\\n"; >> gpu_monitor.cpp
echo     std::cout ^<^< "📊 Operaciones/segundo: " ^<^< ^(10000.0 / duration^) ^<^< "\\n"; >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     // Check memory usage >> gpu_monitor.cpp
echo     size_t free_mem, total_mem; >> gpu_monitor.cpp
echo     cudaMemGetInfo^(&free_mem, &total_mem^); >> gpu_monitor.cpp
echo     std::cout ^<^< "💾 Memoria GPU usada: " ^<^< ^(total_mem - free_mem^) / ^(1024*1024^) ^<^< " MB\\n"; >> gpu_monitor.cpp
echo     std::cout ^<^< "💾 Memoria GPU libre: " ^<^< free_mem / ^(1024*1024^) ^<^< " MB\\n\\n"; >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     cublasDestroy^(handle^); >> gpu_monitor.cpp
echo     cudaFree^(d_A^); >> gpu_monitor.cpp
echo     cudaFree^(d_B^); >> gpu_monitor.cpp
echo     cudaFree^(d_C^); >> gpu_monitor.cpp
echo } >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo int main() { >> gpu_monitor.cpp
echo     std::cout ^<^< "🚀 VERIFICADOR DE GPU NVIDIA DEDICADA\\n"; >> gpu_monitor.cpp
echo     std::cout ^<^< "====================================\\n"; >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     force_nvidia_gpu^(^); >> gpu_monitor.cpp
echo     stress_test_gpu^(^); >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     std::cout ^<^< "\\n✅ Verificación completada.\\n"; >> gpu_monitor.cpp
echo     std::cout ^<^< "\\n🔍 INSTRUCCIONES:\\n"; >> gpu_monitor.cpp
echo     std::cout ^<^< "1. Ejecuta 'nvidia-smi' en otra ventana CMD\\n"; >> gpu_monitor.cpp
echo     std::cout ^<^< "2. Observa el uso de GPU mientras este programa funciona\\n"; >> gpu_monitor.cpp
echo     std::cout ^<^< "3. Debe aparecer uso ^>80%% en tu GPU NVIDIA\\n\\n"; >> gpu_monitor.cpp
echo. >> gpu_monitor.cpp
echo     return 0; >> gpu_monitor.cpp
echo } >> gpu_monitor.cpp

REM Compilar el verificador
cl /O2 /EHsc /DUSE_CUDA /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   gpu_monitor.cpp ^
   /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib /OUT:gpu_monitor.exe

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando verificador
    pause
    exit /b 1
)

echo ✅ Verificador compilado exitosamente

echo.
echo 🚀 4. EJECUTANDO VERIFICADOR...
echo ========================================
echo.
echo ⚠️  IMPORTANTE: Abre otra ventana CMD y ejecuta:
echo    nvidia-smi -l 1
echo    Para monitorear el uso de GPU en tiempo real
echo.
echo Presiona cualquier tecla para continuar...
pause

echo.
echo 🔥 INICIANDO STRESS TEST DE GPU...
gpu_monitor.exe

echo.
echo 🔍 5. VERIFICACIÓN FINAL...
echo ========================================
nvidia-smi

echo.
echo 📊 ANÁLISIS COMPLETADO
echo ========================================
echo Si tu GPU NVIDIA no mostró uso alto ^(^>80%%^):
echo 1. Verifica drivers NVIDIA actualizados
echo 2. Revisa configuración de GPU en Panel de Control NVIDIA
echo 3. Asegúrate que CUDA use GPU dedicada, no integrada
echo.

REM Limpiar archivos temporales
if exist "gpu_monitor.cpp" del "gpu_monitor.cpp"

pause