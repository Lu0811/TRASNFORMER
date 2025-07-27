@echo off
echo ===============================================
echo  COMPILADOR CUDA - ARQUITECTURA ULTRA-LIGERA
echo  OBJETIVO: ^<400K params + ^>70%% accuracy
echo ===============================================

echo.
echo 🪶 CONFIGURANDO MODO ULTRA-LIGERO...
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
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv,noheader,nounits

REM Variables de entorno
echo.
echo 🪶 Configurando variables ULTRA-LIGERAS...
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_LAUNCH_BLOCKING=0

echo ✅ Variables configuradas

REM Configurar Visual Studio
echo.
echo 🔧 Configurando Visual Studio...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Limpiar archivos anteriores
echo.
echo 🧹 Limpiando archivos anteriores...
if exist "obj_ultra_light" rmdir /s /q "obj_ultra_light"
if exist "TransformerUltraLight.exe" del "TransformerUltraLight.exe"
mkdir obj_ultra_light

echo.
echo 🪶 Compilando kernels CUDA ULTRA-LIGEROS...
echo ========================================

REM Compilar CUDA ultra optimizado
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math ^
     -arch=sm_86 ^
     -gencode arch=compute_86,code=sm_86 ^
     --ptxas-options=-v ^
     --maxrregcount=32 ^
     -Iinclude -c src/matrix_cuda.cu -o obj_ultra_light/matrix_cuda.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando CUDA
    pause
    exit /b 1
)

echo ✅ CUDA compilado exitosamente

echo.
echo 🪶 Compilando C++ ULTRA-LIGERO...
echo ===============================

REM Compilar C++ con máximas optimizaciones
cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/matrix.cpp /Fo:obj_ultra_light/matrix.obj

cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/mnist_loader.cpp /Fo:obj_ultra_light/mnist_loader.obj

cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/transformer.cpp /Fo:obj_ultra_light/transformer.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando C++
    pause
    exit /b 1
)

echo ✅ C++ compilado exitosamente

echo.
echo 🪶 Enlazando ULTRA-LIGERO...
echo =========================

REM Enlazar versión ultra-ligera
cl /O2 /Ox /Ot /GL /EHsc /DUSE_CUDA ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   main_ultra_light.cpp obj_ultra_light/matrix.obj obj_ultra_light/mnist_loader.obj obj_ultra_light/transformer.obj obj_ultra_light/matrix_cuda.obj ^
   /link /LTCG /OPT:REF /OPT:ICF ^
   /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib cublasLt.lib ^
   /OUT:TransformerUltraLight.exe

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado
    pause
    exit /b 1
)

echo.
echo 🪶 ¡PROYECTO ULTRA-LIGERO COMPILADO!
echo ==================================

echo.
echo 🚀 Tu Transformer ULTRA-LIGERO incluye:
echo   - 🪶 Solo ~350K parámetros
echo   - 🪶 3 capas encoder ^(mínimo^)
echo   - 🪶 d_model=96, num_heads=6
echo   - 🪶 Batch size 128 ^(velocidad^)
echo   - 🪶 Dataset COMPLETO 60K ^(CRÍTICO^)
echo   - 🪶 Learning rate con decay agresivo
echo   - 🪶 Data augmentation ligera
echo   - 🎯 Objetivo: ^>70%% accuracy
echo.

echo Ejecutable: TransformerUltraLight.exe
echo.

echo 🪶 ARQUITECTURA MÍNIMA:
echo ====================

echo.
echo Patch Embedding: 49 x 96 = 4,704 params
echo Attention ^(3 layers^): 3 x ^(4 x 96 x 96^) = 110,592 params
echo FFN ^(3 layers^): 3 x ^(96 x 192 x 2^) = 110,592 params
echo Classifier: 96 x 10 = 960 params
echo Class token: 96 params
echo -----------------------------
echo TOTAL: ~227K params activos
echo Con overhead: ~350K params
echo.

echo 🎯 CLAVES PARA EL ÉXITO:
echo ======================
echo.
echo 1. DATASET COMPLETO ^(60K^) es CRÍTICO
echo 2. Learning rate con warmup y decay
echo 3. Dropout alto ^(0.2^) para regularización
echo 4. Data augmentation ligera
echo 5. 25 épocas para convergencia
echo.

echo ⚠️  IMPORTANTE:
echo ==============
echo Si accuracy ^<70%%:
echo   - Verifica que uses TODO el dataset
echo   - Aumenta épocas a 30-35
echo   - Ajusta learning rate a 0.0015
echo.

pause