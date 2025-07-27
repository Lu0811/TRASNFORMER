@echo off
echo ===================================
echo  COMPILADOR SIMPLE PARA DEBUG
echo ===================================

REM Usar la versión optimizada que ya funcionaba
echo Compilando version debug con CPU optimizada...

REM Limpiar
if exist "TransformerDebug.exe" del "TransformerDebug.exe"

echo.
echo Compilando con configuración pequeña para debug...
g++ -std=c++17 -O2 -DUSE_CUDA -Iinclude ^
    -o TransformerDebug.exe main_fast.cpp ^
    obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en compilación
    pause
    exit /b 1
)

echo.
echo 🎉 ¡Versión DEBUG compilada!
echo.
echo Configuración:
echo   - Modelo pequeño (64 dim, 2 layers)
echo   - Learning rate bajo (0.0001)
echo   - Batch pequeño (64)
echo   - Debug activado
echo.
echo Para ejecutar:
echo   .\TransformerDebug.exe
echo.
pause
