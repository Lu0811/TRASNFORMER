@echo off
echo ===============================================
echo  COMPILADOR CUDA OPTIMIZADO (Visual Studio)
echo ===============================================

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

echo ⚡ Compilando kernels CUDA optimizados...
nvcc -c src/matrix_cuda.cu -o obj_vs/matrix_cuda.obj -O3 -arch=sm_86

if errorlevel 1 (
    echo ❌ Error compilando CUDA
    pause
    exit /b 1
)
echo ⚡ CUDA compilado

echo Compilando archivos C++ con MSVC...
cl /c /EHsc /O2 /DUSE_CUDA /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" /Foobj_vs/ src/matrix.cpp src/mnist_loader.cpp src/transformer.cpp

if errorlevel 1 (
    echo ❌ Error compilando C++
    pause
    exit /b 1
)
echo ⚡ C++ compilado

echo Enlazando versión optimizada con MSVC...
cl /EHsc /O2 /DUSE_CUDA main_optimized.cpp obj_vs/matrix.obj obj_vs/mnist_loader.obj obj_vs/transformer.obj obj_vs/matrix_cuda.obj /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" /link "/LIBPATH:C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" cudart.lib cublas.lib /OUT:TransformerOptimized.exe

if errorlevel 1 (
    echo ❌ Error enlazando
    pause
    exit /b 1
)

echo.
echo 🔥 ¡PROYECTO CUDA OPTIMIZADO COMPILADO EXITOSAMENTE!
echo 🎯 Tu Transformer optimizado ahora usa:
echo   - GPU: RTX 3050 Laptop
echo   - CUDA: 12.8 con softmax corregido
echo   - Compilador: Visual Studio 2022
echo   - Modelo: 128D, 8 heads, 4 layers
echo   - Optimización máxima
echo.
echo Ejecutable: TransformerOptimized.exe
echo Para ejecutar:
echo   .\TransformerOptimized.exe
echo.
echo 🚀 ¡Listo para entrenar con modelo optimizado!
pause
