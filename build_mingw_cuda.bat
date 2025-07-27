@echo off
echo ===================================
echo  Compilador CUDA Transformer v3.0
echo  (Compatible con MinGW)
echo ===================================

REM Verificar que NVCC esté disponible
nvcc --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Error: NVCC no encontrado. Asegúrate de tener CUDA instalado y en el PATH.
    echo Verifica que CUDA Toolkit esté instalado correctamente.
    pause
    exit /b 1
)

echo ✅ NVCC encontrado
nvcc --version

REM Crear directorio obj si no existe
if not exist "obj" mkdir obj

echo.
echo Compilando operaciones CUDA con MinGW...

REM Usar nvcc con MinGW como host compiler
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math -arch=sm_61 -Xcompiler -I./include -ccbin g++ -c src/matrix_cuda.cu -o obj/matrix_cuda.o

if %ERRORLEVEL% neq 0 (
    echo ⚠️  NVCC con MinGW falló, intentando compilación alternativa...
    echo.
    echo Compilando CUDA como C++ con definiciones CUDA...
    g++ -std=c++17 -O3 -Wall -DUSE_CUDA -DCUDA_FALLBACK -Iinclude -c src/matrix_cuda_fallback.cpp -o obj/matrix_cuda.o
    
    if %ERRORLEVEL% neq 0 (
        echo ❌ Error en compilación alternativa
        pause
        exit /b 1
    )
    echo ✅ Compilación CUDA alternativa exitosa
) else (
    echo ✅ Operaciones CUDA compiladas con NVCC
)

echo.
echo Compilando archivos C++...
g++ -std=c++17 -O3 -Wall -DUSE_CUDA -Iinclude -c src/matrix.cpp -o obj/matrix.o
g++ -std=c++17 -O3 -Wall -DUSE_CUDA -Iinclude -c src/mnist_loader.cpp -o obj/mnist_loader.o
g++ -std=c++17 -O3 -Wall -DUSE_CUDA -Iinclude -c src/transformer.cpp -o obj/transformer.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando archivos C++
    pause
    exit /b 1
)

echo ✅ Archivos C++ compilados

echo.
echo Enlazando proyecto completo...
g++ -std=c++17 -O3 -Wall -DUSE_CUDA -Iinclude -o FashionMNISTTransformer_CUDA.exe main.cpp obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" -lcudart

if %ERRORLEVEL% neq 0 (
    echo ⚠️  Enlazado con ruta CUDA completa falló, intentando enlazado simple...
    g++ -std=c++17 -O3 -Wall -DUSE_CUDA -Iinclude -o FashionMNISTTransformer_CUDA.exe main.cpp obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o -lcudart
    
    if %ERRORLEVEL% neq 0 (
        echo ❌ Error en el enlazado
        pause
        exit /b 1
    )
)

echo ✅ Proyecto compilado exitosamente: FashionMNISTTransformer_CUDA.exe

echo.
echo Compilando prueba CUDA...
g++ -std=c++17 -O3 -Wall -DUSE_CUDA -Iinclude -o test_cuda.exe test_cuda.cpp obj/matrix_cuda.o -lcudart

if %ERRORLEVEL% neq 0 (
    echo ⚠️  Advertencia: No se pudo compilar la prueba CUDA, pero el proyecto principal está listo
) else (
    echo ✅ Prueba CUDA compilada: test_cuda.exe
)

echo.
echo 🎉 ¡Compilación completada!
echo.
echo Archivos generados:
echo   - FashionMNISTTransformer_CUDA.exe (proyecto principal)
if exist test_cuda.exe echo   - test_cuda.exe (prueba CUDA)
echo.
echo Para ejecutar:
echo   .\FashionMNISTTransformer_CUDA.exe
if exist test_cuda.exe echo   .\test_cuda.exe
echo.
pause
