@echo off
echo ===================================
echo  Transformer OPTIMIZADO FINAL
echo  Sin CUDA - Máximo rendimiento CPU
echo ===================================

REM Limpiar todo
if exist "obj" rmdir /s /q "obj"
if exist "TransformerFast.exe" del "TransformerFast.exe"
mkdir obj

echo.
echo 🚀 Compilando operaciones optimizadas...
g++ -std=c++17 -O3 -march=native -DUSE_CUDA -Iinclude -c src/matrix_cuda_fallback.cpp -o obj/matrix_cuda.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando operaciones
    pause
    exit /b 1
)

echo ✅ Operaciones optimizadas compiladas

echo.
echo 🔧 Compilando núcleo del Transformer...
g++ -std=c++17 -O3 -march=native -DUSE_CUDA -Iinclude -c src/matrix.cpp -o obj/matrix.o
g++ -std=c++17 -O3 -march=native -DUSE_CUDA -Iinclude -c src/mnist_loader.cpp -o obj/mnist_loader.o  
g++ -std=c++17 -O3 -march=native -DUSE_CUDA -Iinclude -c src/transformer.cpp -o obj/transformer.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando núcleo
    pause
    exit /b 1
)

echo ✅ Núcleo del Transformer compilado

echo.
echo 🔗 Enlazando proyecto final...
g++ -std=c++17 -O3 -march=native -DUSE_CUDA -Iinclude ^
    -o TransformerFast.exe main_fast.cpp ^
    obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado final
    pause
    exit /b 1
)

echo.
echo 🎉 ¡TRANSFORMER COMPILADO EXITOSAMENTE!
echo.
echo 📊 Especificaciones:
echo   - Ejecutable: TransformerFast.exe
echo   - Optimización: O3 + march=native (máximo rendimiento)
echo   - Batch size: 256 (optimizado para tu sistema)
echo   - Épocas: 2 (prueba rápida)
echo   - Dataset: Subset de 10,000 muestras
echo.
echo 🚀 CARACTERÍSTICAS DE ENTRENAMIENTO RÁPIDO:
echo   - Modelo pequeño pero eficiente (128 dim, 4 capas)
echo   - Operaciones matriciales optimizadas
echo   - Normalización mejorada
echo   - Métricas en tiempo real
echo.
echo ⚡ Para ejecutar entrenamiento:
echo   .\TransformerFast.exe
echo.
echo ⏱️  Tiempo estimado: 2-3 minutos para entrenamiento completo
echo.
pause
