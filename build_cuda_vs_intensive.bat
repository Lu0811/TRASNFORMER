@echo off
echo ===============================================
echo  COMPILADOR CUDA - GPU SATURACION MAXIMA 80%+
echo  VERSION INTENSIVA PARA SATURAR GPU COMPLETA
echo ===============================================

echo.
echo 🔥 CONFIGURANDO PARA SATURAR GPU AL 80%+...
echo ========================================

REM Verificar NVIDIA drivers
nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ NVIDIA drivers no detectados
    pause
    exit /b 1
)

echo ✅ Drivers NVIDIA detectados

REM Mostrar estado inicial GPU
echo.
echo 📊 ESTADO INICIAL GPU:
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits

REM Configurar variables de entorno CRÍTICAS para SATURAR GPU
echo.
echo 🔧 Configurando variables para SATURAR GPU...
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID  
set NVIDIA_VISIBLE_DEVICES=all
set NVIDIA_DRIVER_CAPABILITIES=compute,utility
set CUDA_LAUNCH_BLOCKING=0
set CUDA_CACHE_DISABLE=0
set CUDA_AUTO_BOOST=1

REM Variables para máximo rendimiento GPU
set GPU_FORCE_64BIT_PTR=0
set GPU_MAX_HEAP_SIZE=100
set GPU_MAX_ALLOC_PERCENT=100
set GPU_SINGLE_ALLOC_PERCENT=100

echo ✅ Variables GPU configuradas para saturación máxima

REM Configurar Visual Studio
echo.
echo 🔧 Configurando Visual Studio...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Limpiar archivos anteriores
echo.
echo 🧹 Limpiando archivos anteriores...
if exist "obj_vs" rmdir /s /q "obj_vs"
if exist "TransformerGPU_Intensive.exe" del "TransformerGPU_Intensive.exe"
mkdir obj_vs

echo.
echo 🔥 Compilando kernels CUDA para SATURACIÓN GPU...
echo ========================================

REM Compilar CUDA con FLAGS INTENSIVOS para saturar GPU
nvcc -std=c++17 -O3 -DUSE_CUDA -DFORCE_NVIDIA_GPU -DGPU_INTENSIVE_MODE --use_fast_math ^
     -arch=sm_86 ^
     -gencode arch=compute_86,code=sm_86 ^
     --ptxas-options=-v,--opt-level=3 ^
     --gpu-architecture=sm_86 ^
     --maxrregcount=64 ^
     --ftz=true ^
     --prec-div=false ^
     --prec-sqrt=false ^
     --fmad=true ^
     -Iinclude -c src/matrix_cuda.cu -o obj_vs/matrix_cuda.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando CUDA
    pause
    exit /b 1
)

echo ✅ CUDA compilado con optimizaciones INTENSIVAS

echo.
echo 🔧 Compilando archivos C++ con optimizaciones INTENSIVAS...
echo ========================================

REM Compilar con MÁXIMAS optimizaciones para saturar GPU
cl /O2 /Ox /Ot /Ob2 /Oi /GL /GS- /Gy /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /DGPU_INTENSIVE_MODE ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/matrix.cpp /Fo:obj_vs/matrix.obj

cl /O2 /Ox /Ot /Ob2 /Oi /GL /GS- /Gy /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /DGPU_INTENSIVE_MODE ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/mnist_loader.cpp /Fo:obj_vs/mnist_loader.obj

cl /O2 /Ox /Ot /Ob2 /Oi /GL /GS- /Gy /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /DGPU_INTENSIVE_MODE ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/transformer.cpp /Fo:obj_vs/transformer.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando C++
    pause
    exit /b 1
)

echo ✅ C++ compilado con optimizaciones INTENSIVAS

echo.
echo 🔗 Enlazando con librerías para SATURACIÓN GPU...
echo ========================================

REM Enlazar usando la versión INTENSIVA con todas las librerías GPU
cl /O2 /Ox /Ot /Ob2 /Oi /GL /GS- /Gy /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /DGPU_INTENSIVE_MODE ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   main_fast_gpu_intensive.cpp obj_vs/matrix.obj obj_vs/mnist_loader.obj obj_vs/transformer.obj obj_vs/matrix_cuda.obj ^
   /link /LTCG /OPT:REF /OPT:ICF /INCREMENTAL:NO ^
   /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib cublasLt.lib cusparse.lib curand.lib cusolver.lib cufft.lib ^
   /OUT:TransformerGPU_Intensive.exe

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado
    pause
    exit /b 1
)

echo.
echo 🔥 ¡PROYECTO COMPILADO PARA SATURAR GPU AL 80%+!
echo ================================================

echo.
echo 📊 Tu Transformer INTENSIVO incluye:
echo   - 🔥 Modelo 2M parámetros ^(4x más grande^)
echo   - 🔥 Batch size 128 ^(2x más grande^)  
echo   - 🔥 6 capas encoder ^(50%% más profundo^)
echo   - 🔥 d_model=256, d_ff=512 ^(2x más ancho^)
echo   - 🔥 Stress test paralelo incluido
echo   - 🔥 Múltiples streams CUDA paralelos
echo   - 🔥 Operaciones intensivas continuas
echo   - 🔥 Monitoreo GPU detallado en tiempo real
echo   - 🔥 Variables entorno para máximo rendimiento
echo.

echo Ejecutable: TransformerGPU_Intensive.exe
echo.

echo 🔍 VERIFICACIÓN PRE-EJECUCIÓN:
echo ========================================

REM Crear verificador de configuración intensiva
echo #include ^<iostream^> > intensive_check.cpp
echo #include ^<cuda_runtime.h^> >> intensive_check.cpp
echo #include ^<cublas_v2.h^> >> intensive_check.cpp
echo int main^(^) { >> intensive_check.cpp
echo     int count; cudaGetDeviceCount^(&count^); >> intensive_check.cpp
echo     printf^("🔥 VERIFICACIÓN MODO INTENSIVO\\n"^); >> intensive_check.cpp
echo     printf^("GPUs detectadas: %%d\\n", count^); >> intensive_check.cpp
echo     for^(int i=0; i^<count; i++^) { >> intensive_check.cpp
echo         cudaDeviceProp prop; cudaGetDeviceProperties^(&prop, i^); >> intensive_check.cpp
echo         printf^("GPU %%d: %%s\\n", i, prop.name^); >> intensive_check.cpp
echo         printf^("  Memoria: %%zu MB\\n", prop.totalGlobalMem/1024/1024^); >> intensive_check.cpp
echo         printf^("  Multiprocessors: %%d\\n", prop.multiProcessorCount^); >> intensive_check.cpp
echo         printf^("  Max threads/block: %%d\\n", prop.maxThreadsPerBlock^); >> intensive_check.cpp
echo         if^(prop.totalGlobalMem ^> 2000000000^) { >> intensive_check.cpp
echo             printf^("  🔥 GPU DEDICADA - LISTA PARA SATURACIÓN\\n"^); >> intensive_check.cpp
echo         } else { >> intensive_check.cpp
echo             printf^("  ⚠️  GPU integrada - saturación limitada\\n"^); >> intensive_check.cpp
echo         } >> intensive_check.cpp
echo     } >> intensive_check.cpp
echo     size_t free_mem, total_mem; >> intensive_check.cpp
echo     cudaMemGetInfo^(&free_mem, &total_mem^); >> intensive_check.cpp
echo     printf^("\\n💾 Memoria GPU disponible: %%zu MB\\n", free_mem/1024/1024^); >> intensive_check.cpp
echo     printf^("🎯 CONFIGURADO PARA USAR ^>80%% GPU\\n"^); >> intensive_check.cpp
echo     return 0; >> intensive_check.cpp
echo } >> intensive_check.cpp

cl /EHsc /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" intensive_check.cpp /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" cudart.lib cublas.lib /OUT:intensive_check.exe >nul 2>&1

if exist "intensive_check.exe" (
    intensive_check.exe
    del intensive_check.exe intensive_check.cpp intensive_check.obj >nul 2>&1
)

echo.
echo 🚀 INSTRUCCIONES PARA SATURAR GPU AL 80%+:
echo ==========================================
echo.
echo 1. ABRIR MONITOREO GPU ^(OBLIGATORIO^):
echo    En otra ventana CMD ejecuta:
echo    nvidia-smi -l 1
echo.
echo 2. EJECUTAR PROGRAMA INTENSIVO:
echo    .\TransformerGPU_Intensive.exe
echo.
echo 3. LO QUE DEBERÍAS VER:
echo    ✅ GPU Utilization: ^>80%%
echo    ✅ Power Usage: ^>50W ^(de 67W máximo^)
echo    ✅ Memory Usage: ^>2000MB
echo    ✅ Temperature: ^>60°C
echo    ✅ Performance: P0 ^(máximo rendimiento^)
echo.
echo 4. DURANTE EL ENTRENAMIENTO:
echo    - Stress test inicial para calentar GPU
echo    - Modelo 4x más grande ^(2M parámetros^)
echo    - Batches 2x más grandes ^(128 muestras^)
echo    - Operaciones paralelas continuas
echo    - Monitoreo detallado cada 10 batches
echo.
echo ⚠️  IMPORTANTE:
echo Si aún no ves ^>80%% uso GPU:
echo 1. Panel de Control NVIDIA ^> Configuración 3D
echo 2. Modo de energía: "Prefer maximum performance"
echo 3. Configuración global: GPU NVIDIA de alto rendimiento
echo 4. Aplicar y reiniciar programa
echo.

pause