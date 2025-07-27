# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Fashion-MNIST Transformer Classifier implemented in C++ with optional CUDA support. The project implements a complete Vision Transformer architecture for classifying Fashion-MNIST images (28x28 grayscale images of clothing items) into 10 categories.

## Build Commands

### Standard Build (CMake)
```cmd
# Windows with Visual Studio
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release

# Windows with MinGW
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
make

# Linux/macOS
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Quick Build Scripts (Windows)
```cmd
# Standard build
build.bat

# CUDA-enabled builds
build_cuda.bat          # Standard CUDA build
build_cuda_vs.bat       # Visual Studio CUDA build
build_cuda_mingw.bat    # MinGW CUDA build
build_cuda_optimized.bat # Optimized CUDA build

# Debug builds
build_debug.bat         # Debug build
build_debug_simple.bat  # Simple debug build

# Other builds
build_optimized.bat     # CPU-optimized build
build_test.bat          # Build tests
```

### Running the Program
```cmd
# Windows (after CMake build)
cd build\Release
FashionMNISTTransformer.exe

# Linux/macOS
cd build
./FashionMNISTTransformer

# Direct executables (if using batch scripts)
TransformerCUDA_VS.exe
TransformerOptimized.exe
```

### Running Tests
```cmd
# Matrix operations test
matrix_test.exe

# Simple functionality test
test_simple.exe

# CUDA functionality test (if CUDA enabled)
test_cuda.exe
```

## Architecture Overview

### Core Components

1. **Matrix Operations** (`include/matrix.h`, `src/matrix.cpp`)
   - Custom matrix class with basic operations (add, multiply, transpose)
   - Activation functions: ReLU, Sigmoid, Tanh, GELU, Softmax
   - Layer normalization and Xavier initialization
   - Optional CUDA acceleration (`src/matrix_cuda.cu`)

2. **Data Loading** (`include/mnist_loader.h`, `src/mnist_loader.cpp`)
   - Loads Fashion-MNIST binary format files
   - Converts raw bytes to normalized matrices
   - Handles both images (28x28) and labels

3. **Transformer Architecture** (`include/transformer.h`, `src/transformer.cpp`)
   - Vision Transformer with patch embedding (4x4 patches from 28x28 images)
   - Multi-head self-attention and cross-attention mechanisms
   - Positional encoding (sinusoidal)
   - 6 encoder layers + 6 decoder layers
   - Feed-forward networks with GELU activation
   - Classification head for 10 classes

### Model Configuration (Default)
- Embedding dimension: 256
- Attention heads: 8
- Encoder/Decoder layers: 6 each
- Feed-forward dimension: 1024
- Patch size: 4x4
- Dropout rate: 0.1

### CUDA Support
When enabled, provides GPU acceleration for:
- Matrix multiplication (`matrix_multiply_cuda`)
- Matrix addition (`matrix_add_cuda`)
- Activation functions (ReLU, GELU, Softmax)
- Layer normalization

Enable CUDA with CMake: `-DUSE_CUDA=ON` (default)

## Dataset Requirements

Fashion-MNIST dataset files must be in the project root:
- `train-images-idx3-ubyte` (60,000 training images)
- `train-labels-idx1-ubyte` (60,000 training labels)
- `t10k-images-idx3-ubyte` (10,000 test images)
- `t10k-labels-idx1-ubyte` (10,000 test labels)

Download from: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/

## Key Files to Modify

- **Model parameters**: `main.cpp` - Transformer constructor call
- **Training configuration**: `main.cpp` - `train_with_batches()` parameters
- **Dataset paths**: `main.cpp` - File path strings
- **CUDA kernels**: `src/matrix_cuda.cu`, `include/cuda_ops.cu`
- **Build configuration**: `CMakeLists.txt`

## Python Support Files

- `python_trainer.py` - Alternative Python implementation for comparison
- `python_visualizater.py` - Visualization of training results
- `dashboard.py` - Interactive dashboard for model analysis

## Important Notes

1. The project includes multiple main files for different purposes:
   - `main.cpp` - Standard implementation
   - `main_optimized.cpp` - Performance-optimized version
   - `main_fast.cpp` - Speed-focused implementation
   - `main_debug.cpp` - Debug version with extra logging

2. Backpropagation is currently a placeholder - full gradient computation not implemented

3. Multiple build configurations exist for different compiler/CUDA setups due to Windows compatibility requirements

4. The project generates various output files:
   - `training_history.csv` - Training metrics per epoch
   - `predictions.csv` - Model predictions on test set
   - `test_metrics.csv` - Detailed test metrics
   - Various `.png` files for visualization