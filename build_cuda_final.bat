@echo off
echo ===============================================
echo  COMPILADOR CUDA DEFINITIVO para Windows
echo ===============================================

REM Configurar variables CUDA
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_INC=%CUDA_PATH%\include
set CUDA_LIB=%CUDA_PATH%\lib\x64

echo Configurando entorno Visual Studio...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Limpiar compilaciones anteriores
if exist "obj" rmdir /s /q "obj"
if exist "TransformerCUDA_Real.exe" del "TransformerCUDA_Real.exe"
mkdir obj

echo.
echo ✅ Compilando kernel CUDA...
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math -arch=sm_86 ^
     -I"%CUDA_INC%" -Iinclude ^
     -c src/matrix_cuda.cu -o obj/matrix_cuda.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando CUDA
    pause
    exit /b 1
)

echo ✅ CUDA compilado exitosamente

echo.
echo Compilando archivos C++ con rutas CUDA...
g++ -std=c++17 -O3 -DUSE_CUDA -I"%CUDA_INC%" -Iinclude -c src/matrix.cpp -o obj/matrix.o
g++ -std=c++17 -O3 -DUSE_CUDA -I"%CUDA_INC%" -Iinclude -c src/mnist_loader.cpp -o obj/mnist_loader.o
g++ -std=c++17 -O3 -DUSE_CUDA -I"%CUDA_INC%" -Iinclude -c src/transformer.cpp -o obj/transformer.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando archivos C++
    pause
    exit /b 1
)

echo ✅ Archivos C++ compilados

echo.
echo Enlazando proyecto con librerías CUDA...
g++ -std=c++17 -O3 -DUSE_CUDA -I"%CUDA_INC%" -Iinclude ^
    -o TransformerCUDA_Real.exe main_fast.cpp ^
    obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o ^
    -L"%CUDA_LIB%" -lcudart

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado final
    pause
    exit /b 1
)

echo.
echo 🎉 ¡PROYECTO CUDA COMPILADO EXITOSAMENTE!
echo.
echo 📊 Configuración:
echo   - GPU: RTX 3050 Laptop (sm_86)
echo   - CUDA: 12.8
echo   - Optimización: O3
echo   - Batch size: 256
echo   - Épocas: 2 (para prueba rápida)
echo.
echo Ejecutable: TransformerCUDA_Real.exe
echo.
echo Para ejecutar:
echo   .\TransformerCUDA_Real.exe
echo.
echo 🚀 Tu Transformer ahora usará la GPU para acelerar las operaciones!
echo.
pause
