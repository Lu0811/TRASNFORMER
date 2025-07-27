@echo off
echo Probando compilación paso a paso...

REM Configurar VS
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Test simple de MSVC
echo Probando compilador MSVC...
cl /? >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: MSVC no funciona
    pause
    exit /b 1
)

echo ✅ MSVC funciona

REM Test de includes básicos
echo Probando includes...
echo #include ^<iostream^> > test_temp.cpp
echo #include ^<algorithm^> >> test_temp.cpp
echo #include ^<windows.h^> >> test_temp.cpp
echo int main() { return 0; } >> test_temp.cpp

cl /EHsc test_temp.cpp /Fo:test_temp.obj >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Problema con includes básicos
    pause
    exit /b 1
)

echo ✅ Includes básicos funcionan
del test_temp.cpp test_temp.obj >nul 2>&1

REM Test compilación de matrix.cpp
echo Probando matrix.cpp...
cl /O2 /EHsc /DUSE_CUDA /std:c++17 /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" /c src/matrix.cpp /Fo:matrix_test.obj

if %ERRORLEVEL% neq 0 (
    echo ERROR: No se puede compilar matrix.cpp
    pause
    exit /b 1
)

echo ✅ matrix.cpp compila bien
del matrix_test.obj >nul 2>&1

echo.
echo 🎉 Todos los tests de compilación pasaron
echo Ya puedes ejecutar build_cuda_vs_fixed.bat
pause