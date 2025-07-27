@echo off
echo ===============================================
echo  COMPILADOR CUDA - GPU NVIDIA DEDICADA FORZADA
echo ===============================================

echo.
echo 🎯 FORZANDO USO DE GPU NVIDIA DEDICADA...
echo ========================================

REM Verificar que nvidia-smi funciona
nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ NVIDIA drivers no detectados
    echo    Instala los drivers más recientes de NVIDIA
    pause
    exit /b 1
)

echo ✅ Drivers NVIDIA detectados

REM Mostrar GPUs disponibles
echo.
echo 📊 GPUS DISPONIBLES:
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv,noheader,nounits

echo.
echo 🔧 CONFIGURACIONES PARA FORZAR GPU DEDICADA:
echo ========================================

REM Configurar variables de entorno para FORZAR GPU NVIDIA
echo Configurando variables de entorno...
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set NVIDIA_VISIBLE_DEVICES=all
set NVIDIA_DRIVER_CAPABILITIES=compute,utility

REM Forzar uso de GPU de alta performance
echo Forzando GPU de alta performance...
powershell -Command "& {Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Application]::SetSuspendState('Hibernate', $false, $false)}" >nul 2>&1

echo ✅ Variables configuradas para GPU dedicada

REM Configurar entorno Visual Studio
echo.
echo 🔧 Configurando Visual Studio...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Limpiar compilaciones anteriores
echo.
echo 🧹 Limpiando archivos anteriores...
if exist "obj_vs" rmdir /s /q "obj_vs"
if exist "TransformerCUDA_VS_Forced.exe" del "TransformerCUDA_VS_Forced.exe"
mkdir obj_vs

echo.
echo ✅ Compilando kernels CUDA con GPU FORZADA...
echo ========================================

REM Compilar CUDA con flags específicos para GPU dedicada
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math ^
     -arch=sm_86 ^
     -gencode arch=compute_86,code=sm_86 ^
     --ptxas-options=-v ^
     --gpu-architecture=sm_86 ^
     -Iinclude -c src/matrix_cuda.cu -o obj_vs/matrix_cuda.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando CUDA
    pause
    exit /b 1
)

echo ✅ CUDA compilado con GPU dedicada forzada

echo.
echo 🔧 Compilando archivos C++ optimizados...
echo ========================================

REM Compilar con optimizaciones específicas
cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/matrix.cpp /Fo:obj_vs/matrix.obj

cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/mnist_loader.cpp /Fo:obj_vs/mnist_loader.obj

cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/transformer.cpp /Fo:obj_vs/transformer.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando C++
    pause
    exit /b 1
)

echo ✅ C++ compilado con optimizaciones

echo.
echo 🔗 Enlazando con librerías GPU dedicada...
echo ========================================

REM Enlazar con optimización total y librerías específicas
cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA /DFORCE_NVIDIA_GPU /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   main_fast.cpp obj_vs/matrix.obj obj_vs/mnist_loader.obj obj_vs/transformer.obj obj_vs/matrix_cuda.obj ^
   /link /LTCG /OPT:REF /OPT:ICF ^
   /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib cublasLt.lib cusparse.lib curand.lib ^
   /OUT:TransformerCUDA_VS_Forced.exe

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado
    pause
    exit /b 1
)

echo.
echo 🎉 ¡PROYECTO COMPILADO CON GPU NVIDIA DEDICADA FORZADA!
echo ======================================================

echo.
echo 📊 Tu Transformer FORZADO usa:
echo   - GPU: NVIDIA DEDICADA (forzada)
echo   - CUDA: 12.8 con optimizaciones específicas
echo   - Compilador: Visual Studio 2022 + optimizaciones máximas
echo   - Variables entorno: GPU dedicada forzada
echo   - Arquitectura: sm_86 específica
echo.
echo Ejecutable: TransformerCUDA_VS_Forced.exe
echo.

echo 🔍 VERIFICACIÓN DE GPU:
echo ========================================
echo Ejecutando verificación rápida...

REM Crear verificador rápido
echo #include ^<iostream^> > quick_gpu_check.cpp
echo #include ^<cuda_runtime.h^> >> quick_gpu_check.cpp
echo int main^(^) { >> quick_gpu_check.cpp
echo     int deviceCount; >> quick_gpu_check.cpp
echo     cudaGetDeviceCount^(&deviceCount^); >> quick_gpu_check.cpp
echo     for^(int i=0; i^<deviceCount; i++^) { >> quick_gpu_check.cpp
echo         cudaDeviceProp prop; >> quick_gpu_check.cpp
echo         cudaGetDeviceProperties^(&prop, i^); >> quick_gpu_check.cpp
echo         printf^("GPU %%d: %%s ^(%%zu MB^)\\n", i, prop.name, prop.totalGlobalMem/1024/1024^); >> quick_gpu_check.cpp
echo         if^(prop.totalGlobalMem ^> 2000000000^) printf^("  ⭐ GPU DEDICADA\\n"^); >> quick_gpu_check.cpp
echo     } >> quick_gpu_check.cpp
echo     int activeGPU; cudaGetDevice^(&activeGPU^); >> quick_gpu_check.cpp
echo     printf^("\\n🎯 GPU ACTIVA: %%d\\n", activeGPU^); >> quick_gpu_check.cpp
echo     return 0; >> quick_gpu_check.cpp
echo } >> quick_gpu_check.cpp

cl /EHsc /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" quick_gpu_check.cpp /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" cudart.lib /OUT:quick_gpu_check.exe >nul 2>&1

if exist "quick_gpu_check.exe" (
    quick_gpu_check.exe
    del quick_gpu_check.exe quick_gpu_check.cpp quick_gpu_check.obj >nul 2>&1
)

echo.
echo 🚀 INSTRUCCIONES DE USO:
echo ========================================
echo 1. Ejecuta: .\TransformerCUDA_VS_Forced.exe
echo 2. Abre otra ventana CMD y ejecuta: nvidia-smi -l 1
echo 3. Deberías ver uso alto en tu GPU NVIDIA dedicada
echo.
echo ⚠️  Si aún no ves uso de GPU NVIDIA:
echo 1. Ve a Panel de Control NVIDIA ^> Configuración 3D
echo 2. Cambia a "GPU NVIDIA de alto rendimiento"
echo 3. Reinicia el programa
echo.

pause