#define NOMINMAX  // Evita conflictos con min/max de Windows
#include <windows.h>
#include "include/transformer.h"
#include "include/mnist_loader.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <numeric>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>
#include <climits>

// Para asegurar que std::min y std::max funcionen
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

void print_fashion_mnist_classes() {
    std::cout << "\n=== Fashion-MNIST Classes ===" << std::endl;
    std::cout << "0: T-shirt/top" << std::endl;
    std::cout << "1: Trouser" << std::endl;
    std::cout << "2: Pullover" << std::endl;
    std::cout << "3: Dress" << std::endl;
    std::cout << "4: Coat" << std::endl;
    std::cout << "5: Sandal" << std::endl;
    std::cout << "6: Shirt" << std::endl;
    std::cout << "7: Sneaker" << std::endl;
    std::cout << "8: Bag" << std::endl;
    std::cout << "9: Ankle boot" << std::endl;
    std::cout << "========================\n" << std::endl;
}

std::vector<int> create_shuffled_indices(int size) {
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    return indices;
}

void normalize_images(std::vector<Matrix>& images) {
    double global_mean = 0.0;
    double global_std = 0.0;
    int total_pixels = 0;
    
    // Calculate mean
    for (const auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                global_mean += img.data[i][j];
                total_pixels++;
            }
        }
    }
    global_mean /= total_pixels;
    
    // Calculate std
    for (const auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                double diff = img.data[i][j] - global_mean;
                global_std += diff * diff;
            }
        }
    }
    global_std = sqrt(global_std / total_pixels);
    
    // Normalize
    for (auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                img.data[i][j] = (img.data[i][j] - global_mean) / (global_std + 1e-8);
            }
        }
    }
    
    std::cout << "✅ Global normalization - Mean: " << std::fixed << std::setprecision(4) 
              << global_mean << ", Std: " << global_std << std::endl;
}

void train_with_batches_fixed(Transformer& model, 
                             const std::vector<Matrix>& train_images,
                             const std::vector<int>& train_labels,
                             int epochs, int batch_size, double initial_lr) {
    
    int num_samples = train_images.size();
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    
    std::cout << "\n🚀 === TRAINING WITH FIXED IMPLEMENTATION ===" << std::endl;
    std::cout << "📊 Training samples: " << num_samples << std::endl;
    std::cout << "📦 Batch size: " << batch_size << std::endl;
    std::cout << "🔄 Batches per epoch: " << num_batches << std::endl;
    std::cout << "🎯 Target epochs: " << epochs << std::endl;
    
    // Open CSV for training history
    std::ofstream history_csv("training_history_fixed.csv");
    if (!history_csv.is_open()) {
        std::cerr << "❌ Error opening training_history_fixed.csv" << std::endl;
        return;
    }
    history_csv << "epoch,batch,loss,accuracy,learning_rate,elapsed_time\n";
    
    auto training_start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // Shuffle data each epoch
        std::vector<int> indices = create_shuffled_indices(num_samples);
        
        double epoch_loss = 0.0;
        double epoch_accuracy = 0.0;
        
        // Improved learning rate scheduling
        double current_lr;
        if (epoch < 3) {
            // Warmup phase
            current_lr = initial_lr * (epoch + 1) / 3.0;
        } else if (epoch < epochs * 0.7) {
            // Stable phase
            current_lr = initial_lr;
        } else {
            // Decay phase
            double progress = (epoch - epochs * 0.7) / (epochs * 0.3);
            current_lr = initial_lr * 0.5 * (1.0 + cos(3.14159 * progress));
        }
        
        std::cout << "\n📈 --- Epoch " << (epoch + 1) << "/" << epochs 
                  << " (LR: " << std::scientific << std::setprecision(2) << current_lr << ") ---" << std::endl;
        
        for (int batch = 0; batch < num_batches; batch++) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            std::vector<Matrix> batch_images;
            std::vector<int> batch_labels;
            
            // Create batch
            int start_idx = batch * batch_size;
            int end_idx = (std::min)(start_idx + batch_size, num_samples);
            int actual_batch_size = end_idx - start_idx;
            
            batch_images.reserve(actual_batch_size);
            batch_labels.reserve(actual_batch_size);
            
            for (int i = start_idx; i < end_idx; i++) {
                batch_images.push_back(train_images[indices[i]]);
                batch_labels.push_back(train_labels[indices[i]]);
            }
            
            // Train batch with FIXED implementation
            std::pair<double, double> batch_result = model.train_batch(batch_images, batch_labels, current_lr);
            double batch_loss = batch_result.first;
            double batch_acc = batch_result.second;
            
            epoch_loss += batch_loss * actual_batch_size;
            epoch_accuracy += batch_acc * actual_batch_size;
            
            auto batch_end = std::chrono::high_resolution_clock::now();
            auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start);
            
            // Progress every 10 batches
            if (batch % 10 == 0 || batch == num_batches - 1) {
                std::cout << "  📦 Batch " << std::setw(3) << (batch + 1) << "/" << num_batches 
                          << " | Loss: " << std::fixed << std::setprecision(4) << batch_loss
                          << " | Acc: " << std::setprecision(1) << (batch_acc * 100.0) << "%"
                          << " | Time: " << batch_duration.count() << "ms" << std::endl;
            }
            
            // Save batch metrics
            history_csv << (epoch + 1) << "," << (batch + 1) << "," 
                       << batch_loss << "," << batch_acc << "," 
                       << current_lr << "," << batch_duration.count() << "\n";
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
        
        double avg_loss = epoch_loss / num_samples;
        double avg_accuracy = (epoch_accuracy / num_samples) * 100.0;
        
        std::cout << "🏁 Epoch " << (epoch + 1) << " Summary: "
                  << epoch_duration.count() << "s | Loss: " 
                  << std::fixed << std::setprecision(5) << avg_loss
                  << " | Acc: " << std::fixed << std::setprecision(2) << avg_accuracy << "%" << std::endl;
        
        // Early stopping conditions
        if (avg_accuracy > 90.0) {
            std::cout << "🎯 Early stopping: Excellent accuracy reached!" << std::endl;
            break;
        }
        
        if (epoch > 5 && avg_accuracy < 30.0) {
            std::cout << "⚠️ Early stopping: Model not learning effectively" << std::endl;
            break;
        }
        
        // Force flush output
        std::cout.flush();
    }
    
    auto training_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::minutes>(training_end - training_start);
    
    std::cout << "\n⏱️ Total training time: " << total_duration.count() << " minutes" << std::endl;
    
    history_csv.close();
}

void evaluate_model_fixed(Transformer& model, 
                         const std::vector<Matrix>& test_images,
                         const std::vector<int>& test_labels,
                         int num_samples = 1000) {
    
    std::cout << "\n🧪 === EVALUATING FIXED MODEL ===" << std::endl;
    
    model.set_training(false);
    
    auto test_start = std::chrono::high_resolution_clock::now();
    
    int correct = 0;
    double total_loss = 0.0;
    
    num_samples = (std::min)(num_samples, static_cast<int>(test_images.size()));
    
    // Class-wise accuracy tracking
    std::vector<int> class_correct(10, 0);
    std::vector<int> class_total(10, 0);
    
    // Vectors for CSV output
    std::vector<int> true_labels_vec;
    std::vector<int> predicted_labels_vec;
    std::vector<double> confidences_vec;
    
    std::cout << "🔍 Testing " << num_samples << " samples..." << std::endl;
    
    for (int i = 0; i < num_samples; i++) {
        if (i % 100 == 0) {
            std::cout << "  Progress: " << i << "/" << num_samples 
                      << " (" << std::fixed << std::setprecision(1) 
                      << (100.0 * i / num_samples) << "%)\r" << std::flush;
        }
        
        Matrix predictions = model.forward(test_images[i]);
        
        // Compute loss
        std::vector<int> single_label = {test_labels[i]};
        total_loss += model.compute_loss(predictions, single_label);
        
        // Get prediction
        int predicted_class = 0;
        double max_prob = predictions.data[0][0];
        for (int j = 1; j < 10; j++) {
            if (predictions.data[0][j] > max_prob) {
                max_prob = predictions.data[0][j];
                predicted_class = j;
            }
        }
        
        // Update statistics
        class_total[test_labels[i]]++;
        if (predicted_class == test_labels[i]) {
            correct++;
            class_correct[test_labels[i]]++;
        }
        
        // Store for CSV
        true_labels_vec.push_back(test_labels[i]);
        predicted_labels_vec.push_back(predicted_class);
        confidences_vec.push_back(max_prob);
    }
    
    auto test_end = std::chrono::high_resolution_clock::now();
    auto test_duration = std::chrono::duration_cast<std::chrono::seconds>(test_end - test_start);
    
    double test_loss = total_loss / num_samples;
    double test_accuracy = (double)correct / num_samples * 100.0;
    
    std::cout << "\n\n📊 === RESULTS === " << std::endl;
    std::cout << "⏱️ Testing completed in " << test_duration.count() << " seconds" << std::endl;
    std::cout << "📉 Test Loss: " << std::fixed << std::setprecision(5) << test_loss << std::endl;
    std::cout << "🎯 Test Accuracy: " << std::fixed << std::setprecision(2) << test_accuracy << "%" << std::endl;
    
    // Per-class accuracy
    std::cout << "\n📋 Per-class Accuracy:" << std::endl;
    std::vector<std::string> class_names = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };
    
    for (int c = 0; c < 10; c++) {
        if (class_total[c] > 0) {
            double class_acc = (double)class_correct[c] / class_total[c] * 100.0;
            std::cout << "  " << c << " (" << std::setw(12) << class_names[c] << "): "
                      << std::fixed << std::setprecision(1) << class_acc << "% "
                      << "(" << class_correct[c] << "/" << class_total[c] << ")" << std::endl;
        }
    }
    
    // Save detailed results
    std::ofstream predictions_csv("predictions_fixed.csv");
    if (predictions_csv.is_open()) {
        predictions_csv << "sample_id,true_label,predicted_label,confidence,correct\n";
        for (int i = 0; i < num_samples; i++) {
            predictions_csv << i << "," << true_labels_vec[i] << "," 
                           << predicted_labels_vec[i] << "," << confidences_vec[i] << ","
                           << (true_labels_vec[i] == predicted_labels_vec[i] ? 1 : 0) << "\n";
        }
        predictions_csv.close();
        std::cout << "💾 Detailed predictions saved to: predictions_fixed.csv" << std::endl;
    }
    
    // Save summary metrics
    std::ofstream metrics_csv("test_metrics_fixed.csv");
    if (metrics_csv.is_open()) {
        metrics_csv << "metric,value\n";
        metrics_csv << "test_loss," << test_loss << "\n";
        metrics_csv << "test_accuracy," << test_accuracy << "\n";
        metrics_csv << "test_samples," << num_samples << "\n";
        metrics_csv << "test_time_seconds," << test_duration.count() << "\n";
        metrics_csv.close();
        std::cout << "📈 Test metrics saved to: test_metrics_fixed.csv" << std::endl;
    }
    
    model.set_training(true);
}

void show_sample_predictions(Transformer& model,
                           const std::vector<Matrix>& images,
                           const std::vector<int>& labels,
                           int num_samples = 5) {
    
    std::vector<std::string> class_names = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };
    
    std::cout << "\n🔍 === SAMPLE PREDICTIONS ===" << std::endl;
    
    model.set_training(false);
    
    for (int i = 0; i < num_samples && i < images.size(); i++) {
        Matrix predictions = model.forward(images[i]);
        
        int predicted_class = 0;
        double max_prob = predictions.data[0][0];
        for (int j = 1; j < 10; j++) {
            if (predictions.data[0][j] > max_prob) {
                max_prob = predictions.data[0][j];
                predicted_class = j;
            }
        }
        
        bool correct = (predicted_class == labels[i]);
        std::string status = correct ? "✅" : "❌";
        
        std::cout << "Sample " << (i + 1) << ": " << status << std::endl;
        std::cout << "  True: " << class_names[labels[i]] << " (" << labels[i] << ")" << std::endl;
        std::cout << "  Predicted: " << class_names[predicted_class] << " (" << predicted_class << ")" << std::endl;
        std::cout << "  Confidence: " << std::fixed << std::setprecision(2) << (max_prob * 100.0) << "%" << std::endl;
        std::cout << std::endl;
    }
    
    model.set_training(true);
}

int main() {
    std::cout << "🔥 === FASHION-MNIST TRANSFORMER (FIXED IMPLEMENTATION) ===" << std::endl;
    std::cout << "🚀 With Complete Backpropagation & CUDA Optimization" << std::endl;
    
    print_fashion_mnist_classes();
    
    // File paths
    std::string train_images_path = "train-images-idx3-ubyte";
    std::string train_labels_path = "train-labels-idx1-ubyte";
    std::string test_images_path = "t10k-images-idx3-ubyte";
    std::string test_labels_path = "t10k-labels-idx1-ubyte";
    
#ifdef USE_CUDA
    std::cout << "⚡ CUDA acceleration: ENABLED" << std::endl;
#else
    std::cout << "🐌 CUDA acceleration: DISABLED" << std::endl;
#endif
    
    try {
        std::cout << "\n📂 Loading Fashion-MNIST dataset..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        MNISTLoader loader;
        std::vector<Matrix> train_images = loader.load_images(train_images_path);
        std::vector<int> train_labels = loader.load_labels(train_labels_path);
        std::vector<Matrix> test_images = loader.load_images(test_images_path);
        std::vector<int> test_labels = loader.load_labels(test_labels_path);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::cout << "✅ Data loading completed in " << duration.count() << " seconds." << std::endl;
        
        // Use substantial amount of data for real training
        int train_samples = (std::min)(10000, static_cast<int>(train_images.size())); // More data
        int test_samples = (std::min)(2000, static_cast<int>(test_images.size()));
        
        train_images.resize(train_samples);
        train_labels.resize(train_samples);
        test_images.resize(test_samples);
        test_labels.resize(test_samples);
        
        std::cout << "\n📊 Dataset Summary:" << std::endl;
        std::cout << "  📚 Training samples: " << train_samples << std::endl;
        std::cout << "  🧪 Test samples: " << test_samples << std::endl;
        
        // Normalize data
        std::cout << "\n🔧 Normalizing images..." << std::endl;
        normalize_images(train_images);
        normalize_images(test_images);
        
        // Create FIXED transformer model
        std::cout << "\n🧠 Initializing FIXED Transformer model..." << std::endl;
        Transformer model(
            256,  // d_model - Increased for better capacity
            8,    // num_heads - True multi-head attention
            6,    // num_layers - Deeper for better learning
            1024, // d_ff - Larger feed-forward
            10,   // num_classes
            4,    // patch_size
            0.1   // dropout_rate
        );
        
        model.print_model_info();
        
        // Optimized training parameters
        int epochs = 10;               // More epochs for better convergence
        int batch_size = 128;          // Reasonable batch size
        double learning_rate = 0.0005; // Conservative LR for stability
        
        std::cout << "\n⚙️ Training Configuration:" << std::endl;
        std::cout << "  🔄 Epochs: " << epochs << std::endl;
        std::cout << "  📦 Batch size: " << batch_size << std::endl;
        std::cout << "  📈 Learning rate: " << learning_rate << std::endl;
        
        // Train the FIXED model
        train_with_batches_fixed(model, train_images, train_labels, epochs, batch_size, learning_rate);
        
        // Evaluate on test set
        evaluate_model_fixed(model, test_images, test_labels, test_samples);
        
        // Show sample predictions
        show_sample_predictions(model, test_images, test_labels, 10);
        
        std::cout << "\n🎉 === TRAINING AND TESTING COMPLETED ===" << std::endl;
        
        std::cout << "\n📄 Generated Files:" << std::endl;
        std::cout << "  📊 training_history_fixed.csv - Detailed training metrics" << std::endl;
        std::cout << "  🎯 predictions_fixed.csv - Individual predictions" << std::endl;
        std::cout << "  📈 test_metrics_fixed.csv - Summary test results" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}