@echo off
echo Compilación rápida de transformer_fixed.cpp...

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

echo Compilando solo transformer_fixed.cpp...
cl /O2 /EHsc /DUSE_CUDA /std:c++17 ^
   /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/transformer_fixed.cpp /Fo:transformer_fixed_test.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando transformer_fixed.cpp
    pause
    exit /b 1
)

echo ✅ transformer_fixed.cpp compila correctamente
del transformer_fixed_test.obj

echo Ahora ejecuta build_cuda_vs_fixed.bat
pause