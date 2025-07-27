# Guía de Compilación y Ejecución en WSL

Esta guía explica cómo compilar y ejecutar el proyecto Fashion-MNIST Transformer en WSL (Windows Subsystem for Linux).

## 🚀 Quick Start

```bash
# 1. Entrar a WSL
wsl

# 2. Navegar al directorio del proyecto
cd /mnt/c/Users/Jharold\ Alonso/Downloads/TRASNFORMER

# 3. Dar permisos de ejecución
chmod +x build_wsl.sh

# 4. Compilar
./build_wsl.sh

# 5. Ejecutar
./bin/transformer_balanced
```

## 📋 Requisitos Previos

### En Windows:
- Windows 10 build 21H2+ o Windows 11
- WSL2 instalado (`wsl --install`)
- Driver NVIDIA 470.14+ (si tienes GPU NVIDIA)

### En WSL:
- Ubuntu 20.04/22.04 (recomendado)
- g++ compiler
- CUDA Toolkit (opcional, para GPU)

## 🔧 Instalación Paso a Paso

### 1. Configurar CUDA en WSL (Opcional - Solo GPU NVIDIA)

```bash
# Dar permisos y ejecutar script de configuración
chmod +x setup_wsl_cuda.sh
./setup_wsl_cuda.sh
```

Este script:
- Verifica WSL2 y GPU NVIDIA
- Instala CUDA Toolkit
- Configura variables de entorno
- Verifica la instalación

### 2. Compilar el Proyecto

#### Opción A: Usando el script build_wsl.sh

```bash
chmod +x build_wsl.sh
./build_wsl.sh
```

El script te preguntará qué versión compilar:
1. Standard
2. Speed Max (<500ms/batch)
3. Balanced (75-80% accuracy) ⭐ Recomendado
4. Ultra Light (<400K params)
5. High Initial Accuracy
6. Todas las versiones

#### Opción B: Usando Makefile

```bash
# Compilar todas las versiones
make -f Makefile.wsl all

# Compilar y ejecutar versión balanceada
make -f Makefile.wsl run

# Ver ayuda
make -f Makefile.wsl help
```

### 3. Descargar Dataset (si no existe)

```bash
# Usando make
make -f Makefile.wsl dataset

# O manualmente
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

## 🏃 Ejecutar el Transformer

### Ejecutar directamente:

```bash
# Versión balanceada (recomendada)
./bin/transformer_balanced

# Versión ultra-rápida
./bin/transformer_speed_max

# Versión ultra-ligera
./bin/transformer_ultra_light

# Alta precisión inicial
./bin/transformer_high_acc
```

### Usando el script de ejecución:

```bash
# Ejecutar versión por defecto (balanced)
./run_transformer.sh

# Ejecutar versión específica
./run_transformer.sh transformer_speed_max
```

## 📊 Comparación de Versiones

| Versión | Parámetros | Tiempo/batch | Accuracy | Uso |
|---------|------------|--------------|----------|-----|
| **Standard** | ~3M | ~1500ms | 85-90% | Referencia |
| **Speed Max** | 800K | <500ms | 50-55% | Demos rápidas |
| **Balanced** | 2M | <800ms | 75-80% | ⭐ Recomendado |
| **Ultra Light** | 350K | <300ms | 70-73% | Edge devices |
| **High Acc** | 1.5M | <1000ms | >80% | Mejor inicio |

## 🐛 Solución de Problemas

### 1. "CUDA no encontrado"
```bash
# Verificar si CUDA está instalado
which nvcc

# Si no está instalado, ejecutar:
./setup_wsl_cuda.sh
```

### 2. "Permission denied"
```bash
# Dar permisos de ejecución
chmod +x *.sh
chmod +x bin/*
```

### 3. "Dataset no encontrado"
```bash
# Descargar dataset
make -f Makefile.wsl dataset
```

### 4. GPU no detectada en WSL
- Verifica que tienes Windows 11 o Windows 10 21H2+
- Actualiza drivers NVIDIA en Windows (no en WSL)
- Verifica con: `nvidia-smi.exe` en PowerShell

### 5. Compilación lenta
```bash
# Usar compilación paralela
make -f Makefile.wsl -j$(nproc) all
```

## 🎯 Tips de Rendimiento

### Para máxima velocidad:
```bash
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
./bin/transformer_speed_max
```

### Para mejor accuracy:
```bash
# Usar dataset completo (60K muestras)
./bin/transformer_balanced
```

### Monitorear GPU (desde Windows):
```powershell
# En PowerShell
nvidia-smi -l 1
```

## 📈 Benchmarks

Ejecutar todos los benchmarks:
```bash
make -f Makefile.wsl benchmark
```

## 🔍 Verificar Instalación

```bash
# Verificar compilador
g++ --version

# Verificar CUDA (si aplica)
nvcc --version

# Verificar ejecutables
ls -la bin/

# Test rápido
./bin/transformer_speed_max
```

## 📝 Notas Importantes

1. **WSL2 vs WSL1**: Se requiere WSL2 para soporte de GPU
2. **nvidia-smi**: Puede no funcionar en WSL (es normal)
3. **Rutas**: Usa `/mnt/c/` para acceder a archivos de Windows
4. **Permisos**: Los archivos creados en Windows pueden necesitar chmod
5. **GPU**: La GPU funciona aunque nvidia-smi no esté disponible

## 🆘 Ayuda Adicional

Si encuentras problemas:
1. Verifica la versión de WSL: `wsl --version`
2. Actualiza WSL: `wsl --update`
3. Revisa logs de compilación en `/tmp/compile_error.log`
4. Consulta: https://docs.nvidia.com/cuda/wsl-user-guide/