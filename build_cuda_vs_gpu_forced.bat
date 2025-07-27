@echo off
echo ===============================================
echo  COMPILADOR CUDA - GPU NVIDIA DEDICADA FORZADA
echo  VERSION MEJORADA CON MONITOREO GPU
echo ===============================================

echo.
echo 🎯 CONFIGURANDO PARA GPU NVIDIA DEDICADA...
echo ========================================

REM Verificar NVIDIA drivers
nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ NVIDIA drivers no detectados
    pause
    exit /b 1
)

echo ✅ Drivers NVIDIA detectados

REM Mostrar GPUs disponibles
echo.
echo 📊 GPUS DISPONIBLES:
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv,noheader,nounits

REM Configurar variables de entorno CRÍTICAS para GPU dedicada
echo.
echo 🔧 Configurando variables de entorno GPU...
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID  
set NVIDIA_VISIBLE_DEVICES=all
set NVIDIA_DRIVER_CAPABILITIES=compute,utility
set CUDA_LAUNCH_BLOCKING=0

echo ✅ Variables GPU configuradas

REM Configurar Visual Studio
echo.
echo 🔧 Configurando Visual Studio...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Limpiar archivos anteriores
echo.
echo 🧹 Limpiando archivos anteriores...
if exist "obj_vs" rmdir /s /q "obj_vs"
if exist "TransformerGPU_Forced.exe" del "TransformerGPU_Forced.exe"
mkdir obj_vs

echo.
echo ✅ Compilando kernels CUDA optimizados...
echo ========================================

REM Compilar CUDA con arquitectura específica
nvcc -std=c++17 -O3 -DUSE_CUDA -DFORCE_NVIDIA_GPU --use_fast_math ^
     -arch=sm_86 ^
     -gencode arch=compute_86,code=sm_86 ^
     --ptxas-options=-v ^
     -Iinclude -c src/matrix_cuda.cu -o obj_vs/matrix_cuda.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando CUDA
    pause
    exit /b 1
)

echo ✅ CUDA compilado exitosamente

echo.
echo 🔧 Compilando archivos C++...
echo ========================================

REM Compilar matrix.cpp
cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/matrix.cpp /Fo:obj_vs/matrix.obj

REM Compilar mnist_loader.cpp  
cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/mnist_loader.cpp /Fo:obj_vs/mnist_loader.obj

REM Compilar transformer.cpp
cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/transformer.cpp /Fo:obj_vs/transformer.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando C++
    pause
    exit /b 1
)

echo ✅ C++ compilado exitosamente

echo.
echo 🔗 Enlazando con optimizaciones máximas...
echo ========================================

REM Enlazar usando la versión mejorada con GPU forzada
cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   main_fast_forced_gpu.cpp obj_vs/matrix.obj obj_vs/mnist_loader.obj obj_vs/transformer.obj obj_vs/matrix_cuda.obj ^
   /link /LTCG /OPT:REF /OPT:ICF ^
   /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib cublasLt.lib cusparse.lib curand.lib ^
   /OUT:TransformerGPU_Forced.exe

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado
    pause
    exit /b 1
)

echo.
echo 🎉 ¡PROYECTO COMPILADO CON GPU NVIDIA FORZADA!
echo =============================================

echo.
echo 📊 Tu Transformer GPU FORZADO incluye:
echo   - ✅ GPU NVIDIA DEDICADA (selección automática de la más potente)
echo   - ✅ Monitoreo de uso GPU en tiempo real
echo   - ✅ Detección automática GPU integrada vs dedicada  
echo   - ✅ Variables de entorno optimizadas
echo   - ✅ Compilación con máximas optimizaciones
echo   - ✅ Arquitectura CUDA específica (sm_86)
echo.

echo Ejecutable: TransformerGPU_Forced.exe
echo.

echo 🔍 VERIFICACIÓN RÁPIDA DE GPU:
echo ========================================

REM Crear verificador inline
echo #include ^<iostream^> > gpu_verify.cpp
echo #include ^<cuda_runtime.h^> >> gpu_verify.cpp
echo int main^(^) { >> gpu_verify.cpp
echo     int count; cudaGetDeviceCount^(&count^); >> gpu_verify.cpp
echo     printf^("GPUs detectadas: %%d\\n", count^); >> gpu_verify.cpp
echo     for^(int i=0; i^<count; i++^) { >> gpu_verify.cpp
echo         cudaDeviceProp prop; cudaGetDeviceProperties^(&prop, i^); >> gpu_verify.cpp
echo         printf^("GPU %%d: %%s ^(%%zu MB^)", i, prop.name, prop.totalGlobalMem/1024/1024^); >> gpu_verify.cpp
echo         if^(prop.totalGlobalMem ^> 2000000000^) printf^(" ⭐ DEDICADA"^); >> gpu_verify.cpp
echo         printf^("\\n"^); >> gpu_verify.cpp
echo     } >> gpu_verify.cpp
echo     return 0; >> gpu_verify.cpp
echo } >> gpu_verify.cpp

cl /EHsc /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" gpu_verify.cpp /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" cudart.lib /OUT:gpu_verify.exe >nul 2>&1

if exist "gpu_verify.exe" (
    gpu_verify.exe
    del gpu_verify.exe gpu_verify.cpp gpu_verify.obj >nul 2>&1
)

echo.
echo 🚀 INSTRUCCIONES DE USO:
echo ========================================
echo.
echo 1. EJECUTAR EL PROGRAMA:
echo    .\TransformerGPU_Forced.exe
echo.
echo 2. MONITOREAR GPU (en otra ventana CMD):
echo    nvidia-smi -l 1
echo.
echo 3. LO QUE DEBERÍAS VER:
echo    - Selección automática de GPU más potente
echo    - Uso GPU ^>80%% durante entrenamiento  
echo    - Memoria GPU aumentando progresivamente
echo    - Información detallada de GPU en consola
echo.
echo ⚠️  SI NO VES USO DE GPU NVIDIA:
echo 1. Ve a Panel de Control NVIDIA
echo 2. Configuración 3D ^> Configuración global  
echo 3. Procesador gráfico preferido: GPU NVIDIA de alto rendimiento
echo 4. Aplicar y reiniciar el programa
echo.

pause