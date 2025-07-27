@echo off
echo ===============================================
echo  COMPILADOR OPTIMIZADO PARA ALTA PRECISIÓN
echo ===============================================

REM Configurar entorno Visual Studio
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Limpiar compilaciones anteriores
if exist "obj_vs" rmdir /s /q "obj_vs"
if exist "TransformerOptimized.exe" del "TransformerOptimized.exe"
mkdir obj_vs

echo.
echo ✅ Compilando kernels CUDA con optimizaciones máximas...
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math -arch=sm_86 ^
     -Iinclude -c src/matrix_cuda.cu -o obj_vs/matrix_cuda.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando CUDA
    pause
    exit /b 1
)

echo ✅ CUDA compilado con optimizaciones

echo.
echo Compilando archivos C++ con MSVC optimizado...
cl /O2 /EHsc /DUSE_CUDA /DNOMINMAX /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/matrix.cpp /Fo:obj_vs/matrix.obj

cl /O2 /EHsc /DUSE_CUDA /DNOMINMAX /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/mnist_loader.cpp /Fo:obj_vs/mnist_loader.obj

cl /O2 /EHsc /DUSE_CUDA /DNOMINMAX /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/transformer.cpp /Fo:obj_vs/transformer.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando C++
    pause
    exit /b 1
)

echo ✅ C++ compilado con optimizaciones

echo.
echo Enlazando con optimizaciones máximas...
cl /O2 /EHsc /DUSE_CUDA /DNOMINMAX /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   main_optimized.cpp obj_vs/matrix.obj obj_vs/mnist_loader.obj obj_vs/transformer.obj obj_vs/matrix_cuda.obj ^
   /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib /OUT:TransformerOptimized.exe

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado
    pause
    exit /b 1
)

echo.
echo 🎉 ¡TRANSFORMER OPTIMIZADO COMPILADO EXITOSAMENTE!
echo.
echo 📊 Configuración OPTIMIZADA para alta precisión:
echo   - Modelo: d_model=256, heads=8, layers=6
echo   - Parámetros: ~2.5M (25x más que fast)
echo   - Épocas: 15 (vs 10 en fast)
echo   - Batch size: 64 (optimizado GPU)
echo   - Learning rate: 0.0005 (optimizado)
echo   - Dataset: COMPLETO (60,000 muestras)
echo.
echo 🔥 MEJORAS vs main_fast.cpp:
echo   - Capacidad del modelo: 25x mayor
echo   - Datos de entrenamiento: 6x más
echo   - Épocas: 1.5x más
echo   - Target accuracy: >85%% (vs 67%% actual)
echo.
echo Ejecutable: TransformerOptimized.exe
echo.
echo Para ejecutar:
echo   .\TransformerOptimized.exe
echo.
echo 🚀 ¡Listo para alcanzar >85%% de accuracy!
echo.
pause
