@echo off
echo ===============================================
echo  SOLUCION CUDA para Windows (sin Visual Studio)
echo ===============================================

echo.
echo 1. Descargando Visual Studio Build Tools (solo compilador)...
echo.
echo Ve a este enlace y descarga solo las Build Tools:
echo https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
echo.
echo O alternativamente, vamos a usar CUDA con GCC de otra manera...
echo.

REM Verificar si podemos usar nvcc con --compiler-bindir
echo Intentando compilacion CUDA con GCC personalizado...

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo Método 1: Usando nvcc con GCC forzado...
nvcc --help | findstr "compiler-bindir"

if %ERRORLEVEL% neq 0 (
    echo NVCC no acepta --compiler-bindir, probando método alternativo...
    goto METHOD2
)

echo.
echo Intentando compilar con --compiler-bindir...
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math -arch=sm_86 ^
     --compiler-bindir "C:\mingw64\bin" ^
     -Iinclude -c src/matrix_cuda.cu -o obj/matrix_cuda.o

if %ERRORLEVEL% equ 0 (
    echo ✅ ¡CUDA compilado exitosamente con GCC!
    goto SUCCESS
)

:METHOD2
echo.
echo Método 2: Descargando compilador MSVC standalone...
echo.
echo Necesitas instalar Microsoft C++ Build Tools.
echo.
echo 🔗 Ejecuta este comando en PowerShell como ADMINISTRADOR:
echo.
echo "winget install Microsoft.VisualStudio.2022.BuildTools"
echo.
echo O descarga manualmente desde:
echo https://aka.ms/vs/17/release/vs_buildtools.exe
echo.
echo Después de instalar, ejecuta este script nuevamente.
echo.

REM Verificar si existe alguna instalación de MSVC
if exist "C:\Program Files\Microsoft Visual Studio" (
    echo.
    echo ⚠️  Detecté Visual Studio instalado. Configurando variables...
    
    REM Buscar vcvarsall.bat
    for /r "C:\Program Files\Microsoft Visual Studio" %%i in (vcvarsall.bat) do (
        if exist "%%i" (
            echo Encontré: %%i
            call "%%i" x64
            goto RETRY_NVCC
        )
    )
)

if exist "C:\Program Files (x86)\Microsoft Visual Studio" (
    echo.
    echo ⚠️  Detecté Visual Studio x86 instalado. Configurando variables...
    
    REM Buscar vcvarsall.bat
    for /r "C:\Program Files (x86)\Microsoft Visual Studio" %%i in (vcvarsall.bat) do (
        if exist "%%i" (
            echo Encontré: %%i
            call "%%i" x64
            goto RETRY_NVCC
        )
    )
)

echo.
echo 💡 SOLUCIÓN TEMPORAL:
echo.
echo Mientras instalas Build Tools, puedes usar la versión CPU optimizada
echo que compilamos antes. Es muy rápida también.
echo.
goto END

:RETRY_NVCC
echo.
echo Reintentando compilación CUDA con MSVC...
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math -arch=sm_86 ^
     -Iinclude -c src/matrix_cuda.cu -o obj/matrix_cuda.o

if %ERRORLEVEL% equ 0 (
    echo ✅ ¡CUDA compilado exitosamente con MSVC!
    goto SUCCESS
)

echo ❌ Aún hay problemas. Necesitas instalar Build Tools.
goto END

:SUCCESS
echo.
echo 🎉 ¡CUDA funcionando! Compilando proyecto completo...
echo.

REM Compilar el resto con GCC
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude -c src/matrix.cpp -o obj/matrix.o
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude -c src/mnist_loader.cpp -o obj/mnist_loader.o
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude -c src/transformer.cpp -o obj/transformer.o

REM Enlazar todo
g++ -std=c++17 -O3 -DUSE_CUDA -Iinclude ^
    -o TransformerCUDA_Real.exe main_fast.cpp ^
    obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o ^
    -L"%CUDA_PATH%\lib\x64" -lcudart

if %ERRORLEVEL% equ 0 (
    echo.
    echo 🚀 ¡Proyecto CUDA compilado completamente!
    echo Ejecutable: TransformerCUDA_Real.exe
    echo.
    echo Para ejecutar: .\TransformerCUDA_Real.exe
) else (
    echo ❌ Error en el enlazado final
)

:END
echo.
pause
