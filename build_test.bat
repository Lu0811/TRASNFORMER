@echo off
echo ===============================================
echo  TEST SIMPLE TRANSFORMER
echo ===============================================

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

echo ⚡ Compilando test simple...
nvcc -std=c++17 -DUSE_CUDA -Iinclude ^
     -c src/matrix_cuda.cu -o obj_vs/matrix_cuda_test.obj

cl /EHsc /O2 /DUSE_CUDA /Iinclude ^
   test_simple.cpp ^
   src/matrix.cpp ^
   src/mnist_loader.cpp ^
   src/transformer.cpp ^
   obj_vs/matrix_cuda_test.obj ^
   /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib ^
   /OUT:test_simple.exe

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Error compilando test
    pause
    exit /b 1
)

echo ✅ Test compilado exitosamente!
echo Ejecutando test...
echo.
test_simple.exe

pause
