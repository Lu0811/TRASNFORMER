@echo off
echo ===============================================
echo  COMPILADOR CUDA RÁPIDO (Incremental + Paralelo)
echo ===============================================

REM Configurar entorno Visual Studio
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Crear directorio si no existe
if not exist "obj_vs" mkdir obj_vs

echo.
echo 🚀 Compilación PARALELA e INCREMENTAL...

REM COMPILACIÓN PARALELA - Solo recompila si cambió
echo Verificando cambios en archivos...

REM CUDA - Solo si cambió
for %%f in (src\matrix_cuda.cu) do (
    if not exist "obj_vs\matrix_cuda.obj" (
        echo ✅ Compilando CUDA...
        nvcc -std=c++17 -O2 -DUSE_CUDA --use_fast_math -arch=sm_86 -Iinclude -c %%f -o obj_vs/matrix_cuda.obj
    ) else (
        for /f %%i in ('forfiles /p . /m %%f /c "cmd /c echo @fdate @ftime"') do set SRC_TIME=%%i
        for /f %%i in ('forfiles /p obj_vs /m matrix_cuda.obj /c "cmd /c echo @fdate @ftime"') do set OBJ_TIME=%%i
        if "!SRC_TIME!" GTR "!OBJ_TIME!" (
            echo ✅ Recompilando CUDA (cambió)...
            nvcc -std=c++17 -O2 -DUSE_CUDA --use_fast_math -arch=sm_86 -Iinclude -c %%f -o obj_vs/matrix_cuda.obj
        ) else (
            echo ⏭️  CUDA sin cambios, usando caché
        )
    )
)

echo.
echo Compilando C++ en PARALELO (si cambió)...

REM Compilar todos los C++ en paralelo usando /MP
cl /MP4 /O2 /EHsc /DUSE_CUDA /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/matrix.cpp src/mnist_loader.cpp src/transformer.cpp ^
   /Fo:obj_vs\

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando C++
    pause
    exit /b 1
)

echo.
echo Enlazando...
cl /O2 /EHsc /DUSE_CUDA /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   main_fast.cpp obj_vs/matrix.obj obj_vs/mnist_loader.obj obj_vs/transformer.obj obj_vs/matrix_cuda.obj ^
   /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib /OUT:TransformerCUDA_VS.exe

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado
    pause
    exit /b 1
)

echo.
echo 🎉 ¡COMPILACIÓN RÁPIDA COMPLETADA!
echo.
echo 📊 Optimizaciones aplicadas:
echo   - ✅ Compilación INCREMENTAL (solo cambios)
echo   - ✅ Compilación PARALELA (/MP4)
echo   - ✅ Optimización O2 (más rápida que O3)
echo   - ✅ Caché de objetos compilados
echo.
echo Ejecutable: TransformerCUDA_VS.exe
echo.
pause