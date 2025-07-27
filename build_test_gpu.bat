@echo off
echo Compilando test de GPU...

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

cl /O2 /EHsc /DUSE_CUDA /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   test_gpu.cpp obj_vs/matrix.obj obj_vs/matrix_cuda.obj ^
   /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib /OUT:test_gpu.exe

if %ERRORLEVEL% eq 0 (
    echo ✅ Compilado exitosamente
    echo.
    echo Ejecutando test...
    test_gpu.exe
) else (
    echo ❌ Error compilando
)

pause