// Version simplificada para debug sin dependencias CUDA
#include "include/transformer.h"
#include "include/mnist_loader.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <cstring>

void simple_info() {
    std::cout << "=== DEBUG VERSION TRANSFORMER ===" << std::endl;
    std::cout << "Compiled without CUDA headers for debugging" << std::endl;
    std::cout << "=================================" << std::endl;
}

class SimpleTrainer {
public:
    // Parámetros optimizados para debug rápido
    static const int MODEL_DIM = 64;
    static const int NUM_HEADS = 4;  
    static const int NUM_LAYERS = 2;
    static const int FF_DIM = 128;
    static const int MAX_SEQ_LEN = 784;
    static const int VOCAB_SIZE = 256;
    static const int NUM_CLASSES = 10;
    static const int BATCH_SIZE = 32;  // Reducido para debug
    static const int NUM_EPOCHS = 1;   // Solo 1 época para debug
    
    Transformer* model;
    
    SimpleTrainer() {
        model = new Transformer(MODEL_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, 
                               MAX_SEQ_LEN, VOCAB_SIZE, NUM_CLASSES);
    }
    
    ~SimpleTrainer() {
        delete model;
    }
    
    void train_batch(const std::vector<Matrix>& batch_inputs,
                     const std::vector<int>& batch_labels,
                     float learning_rate) {
        
        std::cout << "\n=== TRAINING BATCH DEBUG ===" << std::endl;
        std::cout << "Batch size: " << batch_inputs.size() << std::endl;
        std::cout << "Learning rate: " << learning_rate << std::endl;
        
        float total_loss = 0.0f;
        int correct = 0;
        
        for (size_t i = 0; i < batch_inputs.size(); ++i) {
            std::cout << "\n--- Sample " << i << " (Label: " << batch_labels[i] << ") ---" << std::endl;
            
            // Forward pass
            Matrix output = model->forward(batch_inputs[i]);
            
            // Verificar dimensiones del output
            std::cout << "Model output dimensions: " << output.get_rows() << "x" << output.get_cols() << std::endl;
            
            // Verificar output del modelo (primeras 5 valores)
            std::cout << "Model output (first row, first 5 values): ";
            for (int j = 0; j < std::min(5, output.get_cols()); ++j) {
                std::cout << std::fixed << std::setprecision(6) << output[0][j] << " ";
            }
            std::cout << std::endl;
            
            // Convertir label individual a vector
            std::vector<int> single_label = {batch_labels[i]};
            
            // Compute loss con debug detallado
            double loss = model->compute_loss(output, single_label);
            std::cout << "Computed loss: " << std::fixed << std::setprecision(6) << loss << std::endl;
            
            total_loss += loss;
            
            // Accuracy - encontrar la predicción con valor máximo
            int prediction = 0;
            double max_val = output[0][0];
            for (int j = 1; j < output.get_cols(); ++j) {
                if (output[0][j] > max_val) {
                    max_val = output[0][j];
                    prediction = j;
                }
            }
            
            if (prediction == batch_labels[i]) {
                correct++;
                std::cout << "✓ Correct prediction: " << prediction << " (confidence: " << max_val << ")" << std::endl;
            } else {
                std::cout << "✗ Wrong prediction: " << prediction << " (expected: " << batch_labels[i] << ", confidence: " << max_val << ")" << std::endl;
            }
            
            // Solo procesar primeras 3 muestras en debug
            if (i >= 2) {
                std::cout << "... (processing remaining " << (batch_inputs.size() - i - 1) << " samples)" << std::endl;
                break;
            }
        }
        
        float avg_loss = total_loss / batch_inputs.size();
        float accuracy = (float)correct / batch_inputs.size();
        
        std::cout << "\n=== BATCH RESULTS ===" << std::endl;
        std::cout << "Average Loss: " << std::fixed << std::setprecision(6) << avg_loss << std::endl;
        std::cout << "Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100 << "%" << std::endl;
        std::cout << "=====================" << std::endl;
    }
};

int main() {
    simple_info();
    
    try {
        // Cargar datos
        std::cout << "\nLoading Fashion-MNIST data..." << std::endl;
        
        MNISTLoader loader;
        auto train_images = loader.load_images("train-images-idx3-ubyte");
        auto train_labels = loader.load_labels("train-labels-idx1-ubyte");
        
        if (train_images.empty() || train_labels.empty()) {
            std::cerr << "Error: No se pudieron cargar los datos de entrenamiento" << std::endl;
            return -1;
        }
        
        std::cout << "Loaded " << train_images.size() << " training samples" << std::endl;
        
        // Crear trainer
        SimpleTrainer trainer;
        
        // Preparar un batch pequeño para debug
        const int debug_batch_size = 8;  // Muy pequeño para debug detallado
        std::vector<Matrix> batch_inputs;
        std::vector<int> batch_labels;
        
        for (int i = 0; i < debug_batch_size && i < train_images.size(); ++i) {
            batch_inputs.push_back(train_images[i]);
            batch_labels.push_back(train_labels[i]);
        }
        
        std::cout << "\n=== DEBUG TRAINING START ===" << std::endl;
        std::cout << "Using " << debug_batch_size << " samples for detailed debugging" << std::endl;
        
        // Entrenar con learning rate muy bajo para debug
        const float debug_lr = 0.00001f;
        trainer.train_batch(batch_inputs, batch_labels, debug_lr);
        
        std::cout << "\n=== DEBUG COMPLETE ===" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
