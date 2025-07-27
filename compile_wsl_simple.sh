#!/bin/bash
# Script simple de compilación para WSL

echo "Compilando Transformer para WSL..."

# Crear directorios
mkdir -p obj_wsl bin

# Compilar archivos C++
echo "Compilando matrix.cpp..."
g++ -std=c++17 -O3 -Iinclude -c src/matrix.cpp -o obj_wsl/matrix.o

echo "Compilando mnist_loader.cpp..."
g++ -std=c++17 -O3 -Iinclude -c src/mnist_loader.cpp -o obj_wsl/mnist_loader.o

echo "Compilando transformer.cpp..."
g++ -std=c++17 -O3 -Iinclude -c src/transformer.cpp -o obj_wsl/transformer.o

echo "Compilando main_speed_balanced.cpp..."
g++ -std=c++17 -O3 -Iinclude -c main_speed_balanced.cpp -o obj_wsl/main_speed_balanced.o

echo "Enlazando..."
g++ -o bin/transformer_balanced obj_wsl/main_speed_balanced.o obj_wsl/matrix.o obj_wsl/mnist_loader.o obj_wsl/transformer.o -lm -lpthread

echo "Compilación completada!"
echo "Ejecutar con: ./bin/transformer_balanced"