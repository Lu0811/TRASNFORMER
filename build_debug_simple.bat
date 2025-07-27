@echo off
echo Compilando version DEBUG del Transformer (sin CUDA)...

REM Compilar sin CUDA para debug
g++ -std=c++17 -Iinclude -O0 -g -DDEBUG -o TransformerDebug main_debug.cpp src/matrix.cpp src/transformer.cpp src/mnist_loader.cpp

if %errorlevel% equ 0 (
    echo.
    echo ✓ Compilacion exitosa!
    echo Ejecutando version debug...
    echo.
    .\TransformerDebug.exe
) else (
    echo.
    echo ✗ Error en la compilacion
    pause
)
