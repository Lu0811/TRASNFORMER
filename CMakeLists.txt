cmake_minimum_required(VERSION 3.10)
project(FashionMNISTTransformer LANGUAGES CXX)

# ==============================
# ✅ OPCIÓN PARA ACTIVAR CUDA
# ==============================
option(USE_CUDA "Enable CUDA support" ON)

# ==============================
# 🔧 CONFIGURACIÓN GLOBAL
# ==============================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ==============================
# 🔍 FLAGS DE COMPILACIÓN
# ==============================
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /W3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
endif()

# ==============================
# 🔍 INCLUDES
# ==============================
include_directories(${CMAKE_SOURCE_DIR}/include)

# ==============================
# 📁 ARCHIVOS FUENTE
# ==============================
set(SOURCES
    main.cpp
    src/matrix.cpp
    src/mnist_loader.cpp
    src/transformer.cpp
)

# ==============================
# 🚀 SOPORTE CUDA
# ==============================
if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 61)  # Para GPU MX250 (Compute Capability 6.1)
    list(APPEND SOURCES src/matrix_cuda.cu)
    list(APPEND SOURCES include/cuda_ops.cu)
    add_definitions(-DUSE_CUDA)  # ✅ define la macro en todos los .cpp
    message(STATUS "✅ CUDA habilitado")
endif()

# ==============================
# 🔨 EJECUTABLE
# ==============================
add_executable(${PROJECT_NAME} ${SOURCES})

# 🔗 LINK A CUDA RUNTIME SI SE USA CUDA
if(USE_CUDA)
    target_link_libraries(${PROJECT_NAME} cudart)
endif()

# ==============================
# ✅ OPENMP (OPCIONAL)
# ==============================
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
    message(STATUS "✅ OpenMP encontrado y activado")
else()
    message(STATUS "⚠️ OpenMP no encontrado - se compilará sin paralelismo en CPU")
endif()

# ==============================
# 🖥️ MENSAJES DE INFORMACIÓN
# ==============================
message(STATUS "📦 Proyecto: ${PROJECT_NAME}")
message(STATUS "🛠️ Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "💡 C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "🔧 Compiler: ${CMAKE_CXX_COMPILER_ID}")
