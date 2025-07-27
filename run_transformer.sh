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
