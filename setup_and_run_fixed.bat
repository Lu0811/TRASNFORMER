@echo off
echo ================================================================
echo  🚀 SETUP COMPLETO TRANSFORMER FIXED CON CUDA 🚀
echo ================================================================

echo.
echo [PASO 1] 🔍 Verificando archivos del dataset...

REM Verificar si existen los archivos del dataset
set "DATASET_OK=1"
if not exist "train-images-idx3-ubyte" set "DATASET_OK=0"
if not exist "train-labels-idx1-ubyte" set "DATASET_OK=0"
if not exist "t10k-images-idx3-ubyte" set "DATASET_OK=0"
if not exist "t10k-labels-idx1-ubyte" set "DATASET_OK=0"

if "%DATASET_OK%"=="0" (
    echo ❌ Archivos del dataset Fashion-MNIST no encontrados
    echo.
    echo 💾 Descargando Fashion-MNIST automáticamente...
    
    REM Crear directorio temporal
    if not exist "temp_download" mkdir temp_download
    
    echo Descargando train-images...
    powershell -Command "Invoke-WebRequest -Uri 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz' -OutFile 'temp_download/train-images.gz'"
    
    echo Descargando train-labels...
    powershell -Command "Invoke-WebRequest -Uri 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz' -OutFile 'temp_download/train-labels.gz'"
    
    echo Descargando test-images...
    powershell -Command "Invoke-WebRequest -Uri 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz' -OutFile 'temp_download/test-images.gz'"
    
    echo Descargando test-labels...
    powershell -Command "Invoke-WebRequest -Uri 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz' -OutFile 'temp_download/test-labels.gz'"
    
    echo.
    echo 📦 Descomprimiendo archivos...
    
    REM Descomprimir usando PowerShell (no requiere 7-zip)
    powershell -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::ExtractToDirectory('temp_download\train-images.gz', '.')" 2>nul
    powershell -Command "$infile = 'temp_download\train-images.gz'; $outfile = 'train-images-idx3-ubyte'; $input = New-Object System.IO.FileStream $infile, ([IO.FileMode]::Open), ([IO.FileAccess]::Read), ([IO.FileShare]::Read); $output = New-Object System.IO.FileStream $outfile, ([IO.FileMode]::Create), ([IO.FileAccess]::Write), ([IO.FileShare]::None); $gzipStream = New-Object System.IO.Compression.GzipStream $input, ([IO.Compression.CompressionMode]::Decompress); $gzipStream.CopyTo($output); $gzipStream.Close(); $output.Close(); $input.Close()"
    
    powershell -Command "$infile = 'temp_download\train-labels.gz'; $outfile = 'train-labels-idx1-ubyte'; $input = New-Object System.IO.FileStream $infile, ([IO.FileMode]::Open), ([IO.FileAccess]::Read), ([IO.FileShare]::Read); $output = New-Object System.IO.FileStream $outfile, ([IO.FileMode]::Create), ([IO.FileAccess]::Write), ([IO.FileShare]::None); $gzipStream = New-Object System.IO.Compression.GzipStream $input, ([IO.Compression.CompressionMode]::Decompress); $gzipStream.CopyTo($output); $gzipStream.Close(); $output.Close(); $input.Close()"
    
    powershell -Command "$infile = 'temp_download\test-images.gz'; $outfile = 't10k-images-idx3-ubyte'; $input = New-Object System.IO.FileStream $infile, ([IO.FileMode]::Open), ([IO.FileAccess]::Read), ([IO.FileShare]::Read); $output = New-Object System.IO.FileStream $outfile, ([IO.FileMode]::Create), ([IO.FileAccess]::Write), ([IO.FileShare]::None); $gzipStream = New-Object System.IO.Compression.GzipStream $input, ([IO.Compression.CompressionMode]::Decompress); $gzipStream.CopyTo($output); $gzipStream.Close(); $output.Close(); $input.Close()"
    
    powershell -Command "$infile = 'temp_download\test-labels.gz'; $outfile = 't10k-labels-idx1-ubyte'; $input = New-Object System.IO.FileStream $infile, ([IO.FileMode]::Open), ([IO.FileAccess]::Read), ([IO.FileShare]::Read); $output = New-Object System.IO.FileStream $outfile, ([IO.FileMode]::Create), ([IO.FileAccess]::Write), ([IO.FileShare]::None); $gzipStream = New-Object System.IO.Compression.GzipStream $input, ([IO.Compression.CompressionMode]::Decompress); $gzipStream.CopyTo($output); $gzipStream.Close(); $output.Close(); $input.Close()"
    
    REM Limpiar archivos temporales
    if exist "temp_download" rmdir /s /q "temp_download"
    
    echo ✅ Dataset descargado y descomprimido
) else (
    echo ✅ Dataset Fashion-MNIST encontrado
)

echo.
echo [PASO 2] 🔧 Compilando Transformer Fixed...
call build_cuda_vs_fixed.bat

if %ERRORLEVEL% neq 0 (
    echo ❌ Error durante la compilación
    pause
    exit /b 1
)

echo.
echo [PASO 3] 🚀 Ejecutando Transformer Fixed...
echo.
echo Iniciando entrenamiento con implementación corregida...
echo ⏱️  Tiempo estimado: 5-15 minutos dependiendo de tu GPU
echo.

if exist "TransformerFixed_CUDA_VS.exe" (
    .\TransformerFixed_CUDA_VS.exe
) else (
    echo ❌ Error: No se encontró el ejecutable TransformerFixed_CUDA_VS.exe
    pause
    exit /b 1
)

echo.
echo ================================================================
echo 🎉 ¡PROCESO COMPLETADO!
echo ================================================================
echo.
echo 📊 Archivos generados:
if exist "training_history_fixed.csv" echo   ✅ training_history_fixed.csv
if exist "predictions_fixed.csv" echo   ✅ predictions_fixed.csv  
if exist "test_metrics_fixed.csv" echo   ✅ test_metrics_fixed.csv
echo.
echo 💡 Para analizar resultados:
echo   - Abre los archivos CSV en Excel o similar
echo   - Compara con resultados de la implementación original
echo.

pause