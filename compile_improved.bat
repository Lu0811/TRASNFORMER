@echo off
echo Compilando Transformer Mejorado...

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

if exist "TransformerImproved.exe" del "TransformerImproved.exe"

echo Compilando CUDA...
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math -arch=sm_86 -Iinclude -c src/matrix_cuda.cu -o matrix_cuda.obj

echo Compilando C++...
cl /O2 /EHsc /DUSE_CUDA /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" /c src/matrix.cpp /Fo:matrix.obj
cl /O2 /EHsc /DUSE_CUDA /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" /c src/mnist_loader.cpp /Fo:mnist_loader.obj
cl /O2 /EHsc /DUSE_CUDA /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" /c src/transformer.cpp /Fo:transformer.obj

echo Enlazando...
cl /O2 /EHsc /DUSE_CUDA /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" main_high_initial_acc.cpp matrix.obj mnist_loader.obj transformer.obj matrix_cuda.obj /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" cudart.lib cublas.lib cublasLt.lib curand.lib /OUT:TransformerImproved.exe

echo.
echo Compilacion completada. Ejecutar: TransformerImproved.exe
pause