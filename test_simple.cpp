#include "include/transformer.h"
#include "include/mnist_loader.h"
#include <iostream>

int main() {
    std::cout << "=== TEST SIMPLE DEL TRANSFORMER ===" << std::endl;
    
    // Crear transformer muy simple
    Transformer transformer(16, 2, 1, 32, 10);
    
    // Cargar solo 100 muestras para test
    auto train_images = MNISTLoader::load_images("train-images-idx3-ubyte");
    auto train_labels = MNISTLoader::load_labels("train-labels-idx1-ubyte");
    
    if (train_images.size() > 100) {
        train_images.resize(100);
        train_labels.resize(100);
    }
    
    std::cout << "Loaded " << train_images.size() << " images" << std::endl;
    
    // Test forward pass en las primeras 10 imágenes
    int correct_before = 0;
    for (int i = 0; i < 10; i++) {
        Matrix pred = transformer.forward(train_images[i]);
        
        // Encontrar clase predicha
        int predicted = 0;
        double max_val = pred.data[0][0];
        for (int j = 1; j < pred.cols; j++) {
            if (pred.data[0][j] > max_val) {
                max_val = pred.data[0][j];
                predicted = j;
            }
        }
        
        if (predicted == train_labels[i]) correct_before++;
        
        std::cout << "Image " << i << ": Label=" << train_labels[i] 
                  << ", Predicted=" << predicted 
                  << ", Max_prob=" << max_val << std::endl;
    }
    
    std::cout << "\nAccuracy ANTES entrenamiento: " << (correct_before * 10.0) << "%" << std::endl;
    
    // Entrenar 1 epoch con las 100 muestras
    std::cout << "\n=== ENTRENANDO 1 ÉPOCA ===" << std::endl;
    
    for (int epoch = 0; epoch < 1; epoch++) {
        for (int batch = 0; batch < 10; batch++) {
            std::vector<Matrix> batch_images;
            std::vector<int> batch_labels;
            
            for (int i = batch * 10; i < (batch + 1) * 10; i++) {
                batch_images.push_back(train_images[i]);
                batch_labels.push_back(train_labels[i]);
            }
            
            auto result = transformer.train_batch(batch_images, batch_labels, 0.0001); // LR MUY PEQUEÑO
            
            if (batch % 5 == 0) {
                std::cout << "Batch " << batch << ": Loss=" << result.first 
                          << ", Acc=" << (result.second * 100.0) << "%" << std::endl;
            }
        }
    }
    
    // Test forward pass después del entrenamiento
    int correct_after = 0;
    for (int i = 0; i < 10; i++) {
        Matrix pred = transformer.forward(train_images[i]);
        
        int predicted = 0;
        double max_val = pred.data[0][0];
        for (int j = 1; j < pred.cols; j++) {
            if (pred.data[0][j] > max_val) {
                max_val = pred.data[0][j];
                predicted = j;
            }
        }
        
        if (predicted == train_labels[i]) correct_after++;
        
        std::cout << "Image " << i << ": Label=" << train_labels[i] 
                  << ", Predicted=" << predicted 
                  << ", Max_prob=" << max_val << std::endl;
    }
    
    std::cout << "\nAccuracy DESPUÉS entrenamiento: " << (correct_after * 10.0) << "%" << std::endl;
    std::cout << "Mejora: " << (correct_after - correct_before) << " predicciones" << std::endl;
    
    return 0;
}
