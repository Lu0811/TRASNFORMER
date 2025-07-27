#include "include/matrix.h"
#include "include/transformer.h"
#include "include/mnist_loader.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

void print_cuda_info() {
    std::cout << "🔥 === TRANSFORMER CUDA OPTIMIZADO ===" << std::endl;
#ifdef USE_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "⚡ CUDA habilitado - " << deviceCount << " dispositivos encontrados" << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "   GPU " << i << ": " << prop.name << std::endl;
        std::cout << "   Memoria: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "   Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "   Multiprocessors: " << prop.multiProcessorCount << std::endl;
    }
    
    cudaSetDevice(0);
    std::cout << "   Dispositivo 0 seleccionado para computación" << std::endl;
#else
    std::cout << "❌ CUDA no está disponible" << std::endl;
#endif
    std::cout << "==========================================" << std::endl;
}

int main() {
    print_cuda_info();
    
    std::cout << "=== Fashion-MNIST Classes ===" << std::endl;
    std::cout << "0: T-shirt/top  1: Trouser    2: Pullover  3: Dress     4: Coat" << std::endl;
    std::cout << "5: Sandal       6: Shirt      7: Sneaker   8: Bag       9: Ankle boot" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // CONFIGURACIÓN OPTIMIZADA PARA ALTA PRECISIÓN (>85%)
    const int epochs = 15;                    // Más épocas para convergencia completa
    const int batch_size = 64;               // Batch size optimizado para GPU
    const double learning_rate = 0.0005;     // Learning rate optimizado
    const int subset_size = 60000;           // Dataset completo
    
    std::cout << "🎯 CONFIGURACIÓN OPTIMIZADA PARA ALTA PRECISIÓN:" << std::endl;
    std::cout << "   - Épocas: " << epochs << " (incrementado para convergencia)" << std::endl;
    std::cout << "   - Batch size: " << batch_size << " (optimizado GPU)" << std::endl;
    std::cout << "   - Learning rate: " << learning_rate << " (optimizado)" << std::endl;
    std::cout << "   - Dataset completo: " << subset_size << " muestras" << std::endl;
    std::cout << "   - Target accuracy: >85%" << std::endl;
    
    // Cargar dataset
    std::cout << "Cargando Fashion-MNIST dataset..." << std::endl;
    auto start_load = std::chrono::high_resolution_clock::now();
    
    MNISTLoader loader;
    auto train_images = loader.load_images("train-images-idx3-ubyte");
    auto train_labels = loader.load_labels("train-labels-idx1-ubyte");
    auto test_images = loader.load_images("t10k-images-idx3-ubyte");
    auto test_labels = loader.load_labels("t10k-labels-idx1-ubyte");
    
    auto end_load = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::seconds>(end_load - start_load).count();
    std::cout << "⚡ Dataset cargado en " << load_time << " segundos." << std::endl;
    
    // Usar dataset completo para máxima precisión
    if (train_images.size() > subset_size) {
        std::cout << "🎯 Usando dataset completo: " << subset_size << " muestras para alta precisión" << std::endl;
        // No redimensionar - usar todo el dataset
    }
    
    std::cout << "Muestras de entrenamiento: " << train_images.size() << std::endl;
    std::cout << "Muestras de prueba: " << test_images.size() << std::endl;
    
    // Normalizar imágenes
    std::cout << "Normalizando imágenes..." << std::endl;
    double train_mean = 0.0, train_std = 0.0;
    for (const auto& img : train_images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                train_mean += img.data[i][j];
            }
        }
    }
    train_mean /= (train_images.size() * 28 * 28);
    
    for (const auto& img : train_images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                train_std += (img.data[i][j] - train_mean) * (img.data[i][j] - train_mean);
            }
        }
    }
    train_std = sqrt(train_std / (train_images.size() * 28 * 28));
    
    // Aplicar normalización
    for (auto& img : train_images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                img.data[i][j] = (img.data[i][j] - train_mean) / train_std;
            }
        }
    }
    
    // Normalizar test con parámetros de train
    for (auto& img : test_images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                img.data[i][j] = (img.data[i][j] - train_mean) / train_std;
            }
        }
    }
    
    std::cout << "⚡ Normalización - Media: " << train_mean << ", Std: " << train_std << std::endl;
    
    // Inicializar modelo optimizado
    std::cout << "Inicializando Transformer optimizado..." << std::endl;
    
    // CONFIGURACIÓN DEL MODELO PARA ALTA PRECISIÓN
    const int d_model = 256;      // Capacidad alta
    const int num_heads = 8;      // Multi-head attention completo
    const int num_layers = 6;     // Profundidad suficiente
    const int d_ff = 512;         // Feed-forward amplio
    const int num_classes = 10;
    const int num_patches = 49;   // 7x7 patches de 4x4
    
    Transformer transformer(d_model, num_heads, num_layers, d_ff, num_classes, num_patches);
    
    std::cout << "Transformer initialized with:" << std::endl;
    std::cout << "- d_model: " << d_model << std::endl;
    std::cout << "- num_heads: " << num_heads << std::endl;
    std::cout << "- num_layers: " << num_layers << std::endl;
    std::cout << "- d_ff: " << d_ff << std::endl;
    std::cout << "- num_patches: " << num_patches << std::endl;
    
    std::cout << "=== CONFIGURACIÓN DEL MODELO PARA ALTA PRECISIÓN ===" << std::endl;
    std::cout << "- d_model: " << d_model << " (4x más que fast - alta capacidad)" << std::endl;
    std::cout << "- num_heads: " << num_heads << " (2x más que fast - mejor atención)" << std::endl;
    std::cout << "- num_layers: " << num_layers << " (3x más que fast - más profundidad)" << std::endl;
    std::cout << "- d_ff: " << d_ff << " (4x más que fast)" << std::endl;
    std::cout << "- classes: " << num_classes << std::endl;
    std::cout << "- Parámetros estimados: ~2.5M (25x más que fast)" << std::endl;
    std::cout << "- Learning rate: " << learning_rate << " (optimizado)" << std::endl;
    std::cout << "- Target accuracy: >85% (vs 67% actual)" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    std::cout << "🔥 === INICIANDO ENTRENAMIENTO OPTIMIZADO ===" << std::endl;
    
    auto start_train = std::chrono::high_resolution_clock::now();
    
    int batches_per_epoch = (train_images.size() + batch_size - 1) / batch_size;
    std::cout << "Batches por época: " << batches_per_epoch << std::endl;
    
    std::vector<double> epoch_losses;
    std::vector<double> epoch_accuracies;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "--- ÉPOCA " << (epoch + 1) << "/" << epochs << " ---" << std::endl;
        
        auto epoch_start = std::chrono::high_resolution_clock::now();
        double total_loss = 0.0;
        double total_accuracy = 0.0;
        
        for (int batch = 0; batch < batches_per_epoch; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, (int)train_images.size());
            
            std::vector<Matrix> batch_images(train_images.begin() + start_idx, train_images.begin() + end_idx);
            std::vector<int> batch_labels(train_labels.begin() + start_idx, train_labels.begin() + end_idx);
            
            auto batch_start = std::chrono::high_resolution_clock::now();
            auto result = transformer.train_batch(batch_images, batch_labels, learning_rate);
            auto batch_end = std::chrono::high_resolution_clock::now();
            
            total_loss += result.first;
            total_accuracy += result.second;
            
            auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start).count();
            
            if (batch % 20 == 0) {  // Mostrar cada 20 batches
                std::cout << "Batch " << batch << "/" << batches_per_epoch 
                          << " | Loss: " << std::fixed << std::setprecision(4) << result.first
                          << " | Acc: " << std::setprecision(1) << result.second * 100 << "%"
                          << " | Tiempo: " << batch_time << "ms"
                          << std::endl;
            }
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_time = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start).count();
        
        double avg_loss = total_loss / batches_per_epoch;
        double avg_accuracy = total_accuracy / batches_per_epoch;
        
        epoch_losses.push_back(avg_loss);
        epoch_accuracies.push_back(avg_accuracy);
        
        std::cout << "⚡ ÉPOCA " << (epoch + 1) << " COMPLETADA:" << std::endl;
        std::cout << "   - Tiempo: " << epoch_time << " segundos" << std::endl;
        std::cout << "   - Loss promedio: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
        std::cout << "   - Accuracy: " << std::setprecision(2) << avg_accuracy * 100 << "%" << std::endl;
        std::cout << "   - Muestras/segundo: " << (int)(subset_size / epoch_time) << std::endl;
        std::cout << std::endl;
    }
    
    auto end_train = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_train - start_train).count();
    
    std::cout << "🔥 === ENTRENAMIENTO COMPLETADO ===" << std::endl;
    std::cout << "Tiempo total: " << total_time << " segundos" << std::endl;
    std::cout << "Velocidad promedio: " << (int)(subset_size * epochs / total_time) << " muestras/segundo" << std::endl;
    std::cout << std::endl;
    
    // Evaluación COMPLETA en test set
    std::cout << "🎯 Evaluación COMPLETA en test set..." << std::endl;
    int test_subset = (int)test_images.size();  // Evaluar dataset completo
    int correct = 0;
    
    for (int i = 0; i < test_subset; i++) {
        if (i % 1000 == 0) {
            std::cout << "Evaluando: " << i << "/" << test_subset << " muestras..." << std::endl;
        }
        Matrix prediction = transformer.forward(test_images[i]);
        
        int predicted_class = 0;
        double max_prob = prediction.data[0][0];
        for (int j = 1; j < num_classes; j++) {
            if (prediction.data[0][j] > max_prob) {
                max_prob = prediction.data[0][j];
                predicted_class = j;
            }
        }
        
        if (predicted_class == test_labels[i]) {
            correct++;
        }
    }
    
    double final_accuracy = (double)correct / test_subset;
    std::cout << "\n🎯 === RESULTADOS FINALES ===" << std::endl;
    std::cout << "✅ Accuracy final en test: " << std::setprecision(2) << final_accuracy * 100 << "%" << std::endl;
    std::cout << "✅ Muestras evaluadas: " << test_subset << " (dataset completo)" << std::endl;
    
    if (final_accuracy >= 0.85) {
        std::cout << "\n🏆 ¡OBJETIVO ALCANZADO! Accuracy >= 85%" << std::endl;
    } else {
        std::cout << "\n⚠️  Accuracy por debajo del objetivo (85%)" << std::endl;
        std::cout << "   Recomendación: Aumentar épocas o ajustar hiperparámetros" << std::endl;
    }
    
    // Mostrar progreso del entrenamiento
    std::cout << "📊 === PROGRESO DEL ENTRENAMIENTO ===" << std::endl;
    for (int i = 0; i < epoch_losses.size(); i++) {
        std::cout << "Época " << (i+1) << ": Loss=" << std::fixed << std::setprecision(4) << epoch_losses[i] 
                  << ", Acc=" << std::setprecision(2) << epoch_accuracies[i] * 100 << "%" << std::endl;
    }
    
    std::cout << "🔥 ¡Entrenamiento CUDA optimizado completado exitosamente!" << std::endl;
    std::cout << "🚀 El modelo ahora tiene " << final_accuracy * 100 << "% de accuracy!" << std::endl;
    
    return 0;
}
