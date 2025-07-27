#!/bin/bash

echo "==============================================="
echo " COMPILADOR TRANSFORMER PARA WSL/LINUX"
echo " Con soporte CUDA para GPU NVIDIA"
echo "==============================================="
echo

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Verificar si estamos en WSL
if [[ ! -f /proc/sys/fs/binfmt_misc/WSLInterop ]]; then
    echo -e "${YELLOW}⚠️  Advertencia: No parece estar ejecutándose en WSL${NC}"
fi

# Verificar CUDA
echo -e "${BLUE}🔍 Verificando instalación de CUDA...${NC}"
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✅ CUDA encontrado${NC}"
    nvcc --version | grep "release" | head -1
    
    # Verificar GPU
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅ nvidia-smi disponible${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo -e "${YELLOW}⚠️  nvidia-smi no encontrado${NC}"
    fi
else
    echo -e "${RED}❌ CUDA no encontrado. Compilando versión CPU...${NC}"
    USE_CUDA=0
fi

# Verificar compilador C++
echo
echo -e "${BLUE}🔍 Verificando compilador C++...${NC}"
if command -v g++ &> /dev/null; then
    echo -e "${GREEN}✅ g++ encontrado${NC}"
    g++ --version | head -1
else
    echo -e "${RED}❌ g++ no encontrado. Instalando...${NC}"
    sudo apt-get update && sudo apt-get install -y g++ build-essential
fi

# Crear directorios
echo
echo -e "${BLUE}📁 Creando directorios...${NC}"
mkdir -p obj_wsl
mkdir -p bin

# Configurar flags de compilación
CXX="g++"
CXXFLAGS="-std=c++17 -O3 -march=native -fopenmp -Wall"
INCLUDES="-Iinclude"
LDFLAGS=""
LIBS="-lm -lpthread"

if command -v nvcc &> /dev/null; then
    USE_CUDA=1
    CUDA_PATH="/usr/local/cuda"
    if [ ! -d "$CUDA_PATH" ]; then
        CUDA_PATH="/usr/lib/cuda"
    fi
    
    NVCC="nvcc"
    NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DUSE_CUDA"
    CUDA_INCLUDES="-I${CUDA_PATH}/include"
    CUDA_LIBS="-L${CUDA_PATH}/lib64 -lcudart -lcublas -lcublasLt -lcurand"
    
    CXXFLAGS="$CXXFLAGS -DUSE_CUDA"
    INCLUDES="$INCLUDES $CUDA_INCLUDES"
    LIBS="$LIBS $CUDA_LIBS"
fi

# Función para compilar con indicador de progreso
compile_with_progress() {
    local source=$1
    local output=$2
    local compiler=$3
    local flags=$4
    
    echo -ne "${BLUE}⏳ Compilando $(basename $source)...${NC}"
    
    if $compiler $flags -c $source -o $output 2> /tmp/compile_error.log; then
        echo -e "\r${GREEN}✅ $(basename $source) compilado${NC}                    "
    else
        echo -e "\r${RED}❌ Error compilando $(basename $source)${NC}"
        cat /tmp/compile_error.log
        exit 1
    fi
}

# Compilar archivos CUDA si está disponible
if [ "$USE_CUDA" -eq 1 ]; then
    echo
    echo -e "${YELLOW}🚀 Compilando kernels CUDA...${NC}"
    compile_with_progress "src/matrix_cuda.cu" "obj_wsl/matrix_cuda.o" "$NVCC" "$NVCCFLAGS $INCLUDES"
fi

# Compilar archivos C++
echo
echo -e "${YELLOW}🔨 Compilando archivos C++...${NC}"

# Compilar librerías
compile_with_progress "src/matrix.cpp" "obj_wsl/matrix.o" "$CXX" "$CXXFLAGS $INCLUDES"
compile_with_progress "src/mnist_loader.cpp" "obj_wsl/mnist_loader.o" "$CXX" "$CXXFLAGS $INCLUDES"
compile_with_progress "src/transformer.cpp" "obj_wsl/transformer.o" "$CXX" "$CXXFLAGS $INCLUDES"

# Menú para seleccionar versión
echo
echo -e "${BLUE}📋 Selecciona la versión a compilar:${NC}"
echo "  1) Standard (main.cpp)"
echo "  2) Speed Max (<500ms/batch, 800K params)"
echo "  3) Balanced (2M params, 75-80% accuracy)"
echo "  4) Ultra Light (<400K params)"
echo "  5) High Initial Accuracy"
echo "  6) Todas las versiones"

read -p "Opción (1-6): " choice

compile_version() {
    local main_file=$1
    local output_name=$2
    local description=$3
    
    echo
    echo -e "${BLUE}🎯 Compilando ${description}...${NC}"
    
    # Compilar main
    compile_with_progress "$main_file" "obj_wsl/$(basename $main_file .cpp).o" "$CXX" "$CXXFLAGS $INCLUDES"
    
    # Enlazar
    echo -ne "${BLUE}⏳ Enlazando ${output_name}...${NC}"
    
    OBJECTS="obj_wsl/$(basename $main_file .cpp).o obj_wsl/matrix.o obj_wsl/mnist_loader.o obj_wsl/transformer.o"
    if [ "$USE_CUDA" -eq 1 ]; then
        OBJECTS="$OBJECTS obj_wsl/matrix_cuda.o"
    fi
    
    if $CXX $CXXFLAGS -o "bin/${output_name}" $OBJECTS $LDFLAGS $LIBS 2> /tmp/link_error.log; then
        echo -e "\r${GREEN}✅ ${output_name} creado${NC}                    "
        chmod +x "bin/${output_name}"
    else
        echo -e "\r${RED}❌ Error enlazando ${output_name}${NC}"
        cat /tmp/link_error.log
        exit 1
    fi
}

# Compilar según selección
case $choice in
    1)
        compile_version "main.cpp" "transformer" "versión estándar"
        ;;
    2)
        compile_version "main_speed_max.cpp" "transformer_speed_max" "Speed Max"
        ;;
    3)
        compile_version "main_speed_balanced.cpp" "transformer_balanced" "Balanced"
        ;;
    4)
        compile_version "main_ultra_light.cpp" "transformer_ultra_light" "Ultra Light"
        ;;
    5)
        compile_version "main_high_initial_acc.cpp" "transformer_high_acc" "High Initial Accuracy"
        ;;
    6)
        compile_version "main.cpp" "transformer" "versión estándar"
        compile_version "main_speed_max.cpp" "transformer_speed_max" "Speed Max"
        compile_version "main_speed_balanced.cpp" "transformer_balanced" "Balanced"
        compile_version "main_ultra_light.cpp" "transformer_ultra_light" "Ultra Light"
        compile_version "main_high_initial_acc.cpp" "transformer_high_acc" "High Initial Accuracy"
        ;;
    *)
        echo -e "${RED}❌ Opción inválida${NC}"
        exit 1
        ;;
esac

# Verificar dataset
echo
echo -e "${BLUE}📊 Verificando dataset Fashion-MNIST...${NC}"

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅ $1 encontrado${NC}"
    else
        echo -e "${RED}❌ $1 no encontrado${NC}"
        return 1
    fi
}

DATASET_OK=1
check_file "train-images-idx3-ubyte" || DATASET_OK=0
check_file "train-labels-idx1-ubyte" || DATASET_OK=0
check_file "t10k-images-idx3-ubyte" || DATASET_OK=0
check_file "t10k-labels-idx1-ubyte" || DATASET_OK=0

if [ $DATASET_OK -eq 0 ]; then
    echo
    echo -e "${YELLOW}⚠️  Dataset incompleto. ¿Descargar ahora? (s/n)${NC}"
    read -p "> " download
    
    if [ "$download" = "s" ] || [ "$download" = "S" ]; then
        echo -e "${BLUE}📥 Descargando dataset...${NC}"
        
        # Crear script de descarga
        cat > download_dataset.sh << 'EOF'
#!/bin/bash
wget -q --show-progress http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget -q --show-progress http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget -q --show-progress http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget -q --show-progress http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

echo "Descomprimiendo..."
gunzip -f *.gz
echo "✅ Dataset descargado"
EOF
        chmod +x download_dataset.sh
        ./download_dataset.sh
        rm download_dataset.sh
    fi
fi

# Resumen final
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} ✅ COMPILACIÓN COMPLETADA${NC}"
echo -e "${GREEN}========================================${NC}"
echo

echo -e "${BLUE}📂 Ejecutables creados en bin/:${NC}"
ls -la bin/ 2>/dev/null | grep transformer | awk '{print "   " $9}'

echo
echo -e "${BLUE}🚀 Para ejecutar:${NC}"
echo "   ./bin/transformer"
echo "   ./bin/transformer_speed_max"
echo "   ./bin/transformer_balanced"
echo "   ./bin/transformer_ultra_light"
echo "   ./bin/transformer_high_acc"

if [ "$USE_CUDA" -eq 1 ]; then
    echo
    echo -e "${GREEN}✅ Compilado con soporte CUDA${NC}"
else
    echo
    echo -e "${YELLOW}⚠️  Compilado solo para CPU${NC}"
fi

# Crear script de ejecución
cat > run_transformer.sh << 'EOF'
#!/bin/bash

# Script para ejecutar el transformer con configuración óptima

# Configurar variables de entorno para CUDA
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# Seleccionar ejecutable
if [ -z "$1" ]; then
    EXEC="./bin/transformer_balanced"
else
    EXEC="./bin/$1"
fi

# Verificar que existe
if [ ! -f "$EXEC" ]; then
    echo "❌ Ejecutable no encontrado: $EXEC"
    echo "Ejecutables disponibles:"
    ls -1 bin/transformer* 2>/dev/null
    exit 1
fi

echo "🚀 Ejecutando: $EXEC"
echo "================================"

# Ejecutar con medición de tiempo
time $EXEC

# Mostrar uso de GPU al final
if command -v nvidia-smi &> /dev/null; then
    echo
    echo "📊 Estado final de GPU:"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
fi
EOF

chmod +x run_transformer.sh

echo
echo -e "${BLUE}💡 Script de ejecución creado: ./run_transformer.sh${NC}"
echo "   Uso: ./run_transformer.sh [nombre_ejecutable]"
echo "   Ejemplo: ./run_transformer.sh transformer_balanced"

echo
echo -e "${GREEN}✨ ¡Listo para entrenar en WSL!${NC}"