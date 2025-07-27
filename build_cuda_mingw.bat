@echo off
echo ========================================
echo  COMPILADOR CUDA CON MINGW FORZADO
echo  Para RTX 3050 Laptop GPU
echo ========================================

REM Verificar GPU
echo Detectando GPU CUDA...
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>nul
if %ERRORLEVEL% neq 0 (
    echo ⚠️  nvidia-smi no disponible, pero continuando...
)

REM Limpiar compilaciones anteriores
if exist "obj" rmdir /s /q "obj"
if exist "TransformerCUDA.exe" del "TransformerCUDA.exe"
mkdir obj

echo.
echo 🔧 Configurando NVCC para usar MinGW...
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
echo CUDA_PATH: %CUDA_PATH%

echo.
echo 🚀 Compilando CUDA con MinGW como host compiler...
REM Usar architecture 86 para RTX 3050 (Ampere)
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math ^
     -gencode arch=compute_86,code=sm_86 ^
     --compiler-options "-O3 -DUSE_CUDA" ^
     -ccbin g++ ^
     -Iinclude ^
     -c src/matrix_cuda.cu -o obj/matrix_cuda.o

if %ERRORLEVEL% neq 0 (
    echo ❌ NVCC con MinGW falló, probando sin especificar host compiler...
    
    nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math ^
         -arch=sm_86 ^
         --compiler-options "/O2" ^
         -Iinclude ^
         -c src/matrix_cuda.cu -o obj/matrix_cuda.o
         
    if %ERRORLEVEL% neq 0 (
        echo ❌ NVCC falló completamente. Verificando instalación...
        echo.
        echo Información de NVCC:
        nvcc --version
        echo.
        echo Variables de entorno CUDA:
        echo CUDA_PATH: %CUDA_PATH%
        echo PATH (parte CUDA): 
        echo %PATH% | findstr CUDA
        pause
        exit /b 1
    )
)

echo ✅ CUDA compilado exitosamente!

echo.
echo Compilando archivos C++ del proyecto...
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude -c src/matrix.cpp -o obj/matrix.o
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude -c src/mnist_loader.cpp -o obj/mnist_loader.o
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude -c src/transformer.cpp -o obj/transformer.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando archivos C++
    pause
    exit /b 1
)

echo ✅ Archivos C++ compilados

echo.
echo 🔗 Enlazando con librerías CUDA...
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude ^
    -o TransformerCUDA.exe main_fast.cpp ^
    obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o ^
    -L"%CUDA_PATH%/lib/x64" ^
    -lcudart -lcublas

if %ERRORLEVEL% neq 0 (
    echo ⚠️  Enlazado con ruta completa falló, probando enlazado simple...
    g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude ^
        -o TransformerCUDA.exe main_fast.cpp ^
        obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o ^
        -lcudart -lcublas
        
    if %ERRORLEVEL% neq 0 (
        echo ❌ Error en enlazado. Probando solo con cudart...
        g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude ^
            -o TransformerCUDA.exe main_fast.cpp ^
            obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o ^
            -lcudart
            
        if %ERRORLEVEL% neq 0 (
            echo ❌ Error en enlazado final
            pause
            exit /b 1
        )
    )
)

echo.
echo 🎉 ¡TRANSFORMER CUDA COMPILADO EXITOSAMENTE!
echo.
echo 🚀 Ejecutable: TransformerCUDA.exe
echo 🎯 GPU Target: RTX 3050 (sm_86)
echo ⚡ Optimizaciones: CUDA + CUBLAS
echo 📊 Configuración: 2 épocas, batch 256
echo.
echo Para ejecutar:
echo   .\TransformerCUDA.exe
echo.
pause
