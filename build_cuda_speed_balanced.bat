@echo off
echo ===============================================
echo  COMPILADOR CUDA - VELOCIDAD + ACCURACY BALANCEADO
echo  OBJETIVO: ^<800ms/batch + 75-80%% accuracy
echo ===============================================

echo.
echo ⚡ CONFIGURANDO MODO BALANCEADO...
echo ==================================

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

REM Variables de entorno
echo.
echo ⚡ Configurando variables BALANCEADAS...
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
if exist "obj_balanced" rmdir /s /q "obj_balanced"
if exist "TransformerBalanced.exe" del "TransformerBalanced.exe"
mkdir obj_balanced

echo.
echo ⚡ Compilando kernels CUDA BALANCEADOS...
echo ======================================

REM Compilar CUDA optimizado
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math ^
     -arch=sm_86 ^
     -gencode arch=compute_86,code=sm_86 ^
     --ptxas-options=-v ^
     --maxrregcount=64 ^
     -Iinclude -c src/matrix_cuda.cu -o obj_balanced/matrix_cuda.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando CUDA
    pause
    exit /b 1
)

echo ✅ CUDA compilado exitosamente

echo.
echo ⚡ Compilando C++ BALANCEADO...
echo =============================

REM Compilar C++ con optimizaciones balanceadas
cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/matrix.cpp /Fo:obj_balanced/matrix.obj

cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/mnist_loader.cpp /Fo:obj_balanced/mnist_loader.obj

cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/transformer.cpp /Fo:obj_balanced/transformer.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando C++
    pause
    exit /b 1
)

echo ✅ C++ compilado exitosamente

echo.
echo ⚡ Enlazando MODO BALANCEADO...
echo ============================

REM Enlazar versión balanceada
cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   main_speed_balanced.cpp obj_balanced/matrix.obj obj_balanced/mnist_loader.obj obj_balanced/transformer.obj obj_balanced/matrix_cuda.obj ^
   /link /LTCG /OPT:REF /OPT:ICF ^
   /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib cublasLt.lib curand.lib ^
   /OUT:TransformerBalanced.exe

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado
    pause
    exit /b 1
)

echo.
echo ⚡ ¡PROYECTO BALANCEADO COMPILADO!
echo ================================

echo.
echo 🚀 Tu Transformer BALANCEADO incluye:
echo   - ⚡ Batch size 48 ^(balanceado^)
echo   - 🎯 Modelo 2M parámetros ^(2.5x más grande^)
echo   - 🎯 d_model=176, num_heads=8, layers=5
echo   - 🎯 49 patches ^(4x4^) vs 16 patches
echo   - 🎯 Dataset 40K muestras ^(67%%^)
echo   - 🎯 Learning rate con warmup + cosine decay
echo   - 🎯 Dropout 0.15 para regularización
echo   - ⚡ Objetivo: ^<800ms/batch
echo   - 🎯 Objetivo: 75-80%% accuracy
echo.

echo Ejecutable: TransformerBalanced.exe
echo.

echo ⚡ VERIFICACIÓN BALANCEADA:
echo ========================

REM Crear verificador balanceado
echo #include ^<iostream^> > balanced_check.cpp
echo #include ^<cuda_runtime.h^> >> balanced_check.cpp
echo int main^(^) { >> balanced_check.cpp
echo     printf^("⚡ VERIFICACIÓN MODO BALANCEADO\\n\\n"^); >> balanced_check.cpp
echo     int count; cudaGetDeviceCount^(&count^); >> balanced_check.cpp
echo     if^(count ^> 0^) { >> balanced_check.cpp
echo         cudaDeviceProp prop; cudaGetDeviceProperties^(&prop, 0^); >> balanced_check.cpp
echo         printf^("GPU: %%s ^(%%zu MB^)\\n", prop.name, prop.totalGlobalMem/1024/1024^); >> balanced_check.cpp
echo         printf^("\\n🎯 CONFIGURACIÓN BALANCEADA:\\n"^); >> balanced_check.cpp
echo         printf^("- Modelo: 2M params ^(balanceado^)\\n"^); >> balanced_check.cpp
echo         printf^("- Batch Size: 48\\n"^); >> balanced_check.cpp
echo         printf^("- Learning Rate: 0.0008 con warmup\\n"^); >> balanced_check.cpp
echo         printf^("- Dataset: 40K muestras\\n"^); >> balanced_check.cpp
echo         printf^("- Patches: 49 ^(4x4^)\\n"^); >> balanced_check.cpp
echo         printf^("\\n✅ LISTO PARA VELOCIDAD + ACCURACY\\n"^); >> balanced_check.cpp
echo     } >> balanced_check.cpp
echo     return 0; >> balanced_check.cpp
echo } >> balanced_check.cpp

cl /O2 /EHsc /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" balanced_check.cpp /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" cudart.lib /OUT:balanced_check.exe >nul 2>&1

if exist "balanced_check.exe" (
    balanced_check.exe
    del balanced_check.exe balanced_check.cpp balanced_check.obj >nul 2>&1
)

echo.
echo 🎯 INSTRUCCIONES MODO BALANCEADO:
echo ==============================
echo.
echo 1. EJECUTAR:
echo    .\TransformerBalanced.exe
echo.
echo 2. MONITOREAR GPU:
echo    nvidia-smi -l 1
echo.
echo 3. RESULTADOS ESPERADOS:
echo    ⚡ Tiempo promedio: 600-800ms/batch
echo    🎯 Accuracy final: 75-80%%
echo    ⚡ GPU uso: 60-75%%
echo    🎯 Convergencia: 15-20 épocas
echo    ⚡ Memoria: ~1500MB
echo.
echo 4. MEJORAS vs SPEED MAX:
echo    - 2.5x más parámetros ^(2M vs 800K^)
echo    - 4x más datos ^(40K vs 10K^)
echo    - 3x más patches ^(49 vs 16^)
echo    - Learning rate adaptativo
echo    - Mejor regularización
echo.
echo 5. SI NECESITAS MÁS ACCURACY:
echo    - Aumenta SUBSET_SIZE a 60000
echo    - Aumenta num_layers a 6
echo    - Aumenta EPOCHS a 40
echo.

pause