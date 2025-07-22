# Makefile para Transformer C++ + CUDA

# Compiladores
CXX = g++
NVCC = nvcc

# Rutas
INCLUDE = -Iinclude
CUDA_LIB = -L/usr/local/cuda/lib64

# Flags
CXXFLAGS = -std=c++17 -DUSE_CUDA $(INCLUDE)
NVCCFLAGS = -c
LDFLAGS = $(CUDA_LIB) -lcudart

# Archivos fuente
SRC_CPP = matrix_test.cpp src/matrix.cpp
SRC_CU = src/matrix_cuda.cu

# Archivos objeto
OBJ_CPP = $(SRC_CPP:.cpp=.o)
OBJ_CU = matrix_cuda.o

# Ejecutable final
TARGET = matrix_test

all: $(TARGET)

$(OBJ_CU): $(SRC_CU)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(TARGET): $(SRC_CPP) src/matrix.cpp $(OBJ_CU)
	$(CXX) $(CXXFLAGS) $(SRC_CPP) src/matrix.cpp $(OBJ_CU) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f *.o src/*.o $(TARGET)
