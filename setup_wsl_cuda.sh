#!/bin/bash

# Script para configurar CUDA en WSL2
# Este script ayuda a instalar y configurar CUDA para usar GPU NVIDIA en WSL

echo "==============================================="
echo " CONFIGURACIÓN DE CUDA PARA WSL2"
echo " Para usar GPU NVIDIA en Windows Subsystem Linux"
echo "==============================================="
echo

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Verificar si es WSL2
echo -e "${BLUE}🔍 Verificando WSL2...${NC}"
if [[ $(uname -r) =~ Microsoft ]]; then
    WSL_VERSION=$(wsl.exe -l -v 2>/dev/null | grep -E "^\*" | awk '{print $4}')
    if [[ "$WSL_VERSION" == "2" ]] || [[ -f /proc/sys/fs/binfmt_misc/WSLInterop ]]; then
        echo -e "${GREEN}✅ WSL2 detectado${NC}"
    else
        echo -e "${RED}❌ Se requiere WSL2 para GPU${NC}"
        echo "   Actualiza con: wsl --set-version <distro> 2"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️  No parece ser WSL${NC}"
fi

# Verificar driver NVIDIA en Windows
echo
echo -e "${BLUE}🔍 Verificando driver NVIDIA en Windows...${NC}"
if powershell.exe -Command "Get-WmiObject Win32_VideoController | Select-String NVIDIA" &>/dev/null; then
    echo -e "${GREEN}✅ GPU NVIDIA detectada en Windows${NC}"
    
    # Intentar obtener versión del driver
    DRIVER_VERSION=$(powershell.exe -Command "(Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like '*NVIDIA*'}).DriverVersion" 2>/dev/null | tr -d '\r\n')
    if [ ! -z "$DRIVER_VERSION" ]; then
        echo "   Driver version: $DRIVER_VERSION"
    fi
else
    echo -e "${RED}❌ No se detecta GPU NVIDIA en Windows${NC}"
    echo "   Instala drivers NVIDIA para Windows primero"
    exit 1
fi

# Verificar si CUDA toolkit ya está instalado
echo
echo -e "${BLUE}🔍 Verificando CUDA Toolkit...${NC}"
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✅ CUDA ya instalado${NC}"
    nvcc --version | grep "release"
    
    echo
    echo -e "${BLUE}¿Reinstalar CUDA? (s/n)${NC}"
    read -p "> " reinstall
    if [[ "$reinstall" != "s" && "$reinstall" != "S" ]]; then
        echo "Saltando instalación de CUDA"
        SKIP_CUDA=1
    fi
fi

# Instalar CUDA Toolkit si es necesario
if [ -z "$SKIP_CUDA" ]; then
    echo
    echo -e "${YELLOW}📥 Instalando CUDA Toolkit para WSL...${NC}"
    
    # Detectar distribución
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
    fi
    
    # Configurar repositorio según distribución
    if [[ "$OS" == "ubuntu" ]]; then
        echo -e "${BLUE}Configurando para Ubuntu $VER...${NC}"
        
        # Limpiar instalaciones previas
        sudo apt-get remove --purge -y nvidia-* cuda-* 2>/dev/null
        sudo apt-get autoremove -y
        
        # Instalar dependencias
        sudo apt-get update
        sudo apt-get install -y wget gnupg2 software-properties-common
        
        # Agregar repositorio CUDA
        wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
        sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
        
        # Agregar clave
        sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/7fa2af80.pub
        
        # Agregar repositorio
        sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
        sudo apt-get update
        
        # Instalar CUDA
        echo -e "${YELLOW}Instalando CUDA (esto puede tardar varios minutos)...${NC}"
        sudo apt-get install -y cuda-toolkit-12-3
        
    else
        echo -e "${RED}❌ Distribución no soportada automáticamente${NC}"
        echo "   Visita: https://docs.nvidia.com/cuda/wsl-user-guide/index.html"
        exit 1
    fi
fi

# Configurar variables de entorno
echo
echo -e "${BLUE}🔧 Configurando variables de entorno...${NC}"

CUDA_PATH="/usr/local/cuda"
if [ ! -d "$CUDA_PATH" ]; then
    CUDA_PATH="/usr/lib/cuda"
fi

# Agregar a bashrc si no existe
if ! grep -q "CUDA_PATH" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA configuration for WSL" >> ~/.bashrc
    echo "export CUDA_PATH=$CUDA_PATH" >> ~/.bashrc
    echo "export PATH=\$CUDA_PATH/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo -e "${GREEN}✅ Variables agregadas a ~/.bashrc${NC}"
fi

# Aplicar cambios
export CUDA_PATH=$CUDA_PATH
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Verificar instalación
echo
echo -e "${BLUE}🔍 Verificando instalación...${NC}"

# Verificar nvcc
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✅ nvcc disponible${NC}"
    nvcc --version | grep "release"
else
    echo -e "${RED}❌ nvcc no encontrado${NC}"
fi

# Verificar nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ nvidia-smi disponible${NC}"
    echo
    nvidia-smi
else
    echo -e "${YELLOW}⚠️  nvidia-smi no disponible${NC}"
    echo "   Esto es normal en WSL. La GPU funciona sin nvidia-smi"
fi

# Test de CUDA
echo
echo -e "${BLUE}🧪 Probando CUDA...${NC}"

# Crear programa de prueba
cat > test_cuda_wsl.cu << 'EOF'
#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cout << "❌ Error CUDA: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "✅ CUDA dispositivos encontrados: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "\nGPU " << i << ": " << prop.name << std::endl;
        std::cout << "  Memoria: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    }
    
    return 0;
}
EOF

# Compilar y ejecutar test
if command -v nvcc &> /dev/null; then
    echo -e "${BLUE}Compilando test...${NC}"
    if nvcc -o test_cuda_wsl test_cuda_wsl.cu 2>/dev/null; then
        echo -e "${BLUE}Ejecutando test...${NC}"
        ./test_cuda_wsl
        rm -f test_cuda_wsl test_cuda_wsl.cu
    else
        echo -e "${RED}❌ Error compilando test${NC}"
        rm -f test_cuda_wsl.cu
    fi
fi

# Instrucciones finales
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} CONFIGURACIÓN COMPLETADA${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${BLUE}📋 Próximos pasos:${NC}"
echo "1. Reinicia tu terminal o ejecuta: source ~/.bashrc"
echo "2. Compila el proyecto con: ./build_wsl.sh"
echo "3. O usa make: make -f Makefile.wsl all"
echo
echo -e "${YELLOW}⚠️  Notas importantes para WSL:${NC}"
echo "- nvidia-smi puede no funcionar (es normal)"
echo "- La GPU funciona aunque nvidia-smi no esté disponible"
echo "- Asegúrate de tener Windows 11 o Windows 10 build 21H2+"
echo "- Necesitas NVIDIA Driver 470.14+ en Windows"
echo
echo -e "${BLUE}🔗 Más información:${NC}"
echo "https://docs.nvidia.com/cuda/wsl-user-guide/index.html"