@echo off
echo ===================================
echo  Compilador Transformer SIMPLE
echo  Máximo rendimiento sin complicaciones
echo ===================================

REM Limpiar compilaciones anteriores
if exist "obj" rmdir /s /q "obj"
if exist "TransformerSimple.exe" del "TransformerSimple.exe"
mkdir obj

echo.
echo Compilando implementación CPU optimizada...
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude -c src/matrix_cuda_fallback.cpp -o obj/matrix_cuda.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando operaciones
    pause
    exit /b 1
)

echo ✅ Operaciones compiladas

echo.
echo Compilando archivos del proyecto...
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
echo Enlazando proyecto...
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude ^
    -o TransformerSimple.exe main_fast.cpp ^
    obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado
    pause
    exit /b 1
)

echo.
echo 🚀 ¡Transformer compilado exitosamente!
echo.
echo Ejecutable: TransformerSimple.exe
echo.
echo 📊 Optimizaciones incluidas:
echo   - Compilación O3 (máxima optimización)
echo   - Operaciones matriciales optimizadas
echo   - Batch processing de 256
echo   - Solo 2 épocas para prueba rápida
echo   - Subset de datos para velocidad
echo.
echo Para ejecutar:
echo   .\TransformerSimple.exe
echo.
pause
