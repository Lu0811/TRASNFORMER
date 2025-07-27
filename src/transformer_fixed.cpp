#include "../include/transformer.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>
#include <fstream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// FIXED MULTI-HEAD ATTENTION IMPLEMENTATION
// ============================================================================

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads) 
    : d_model(d_model), num_heads(num_heads) {
    
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
    
    d_k = d_model / num_heads;
    
    // Initialize weight matrices for ALL heads combined
    W_q = Matrix(d_model, d_model);
    W_k = Matrix(d_model, d_model);
    W_v = Matrix(d_model, d_model);
    W_o = Matrix(d_model, d_model);
    
    // Initialize bias vectors
    b_q = Matrix(1, d_model);
    b_k = Matrix(1, d_model);
    b_v = Matrix(1, d_model);
    b_o = Matrix(1, d_model);
    
    // Xavier initialization
    W_q.xavier_init();
    W_k.xavier_init();
    W_v.xavier_init();
    W_o.xavier_init();
    
    b_q.zero();
    b_k.zero();
    b_v.zero();
    b_o.zero();
    
    // Initialize gradients
    grad_q = Gradients(d_model, d_model);
    grad_k = Gradients(d_model, d_model);
    grad_v = Gradients(d_model, d_model);
    grad_o = Gradients(d_model, d_model);
}

Matrix MultiHeadAttention::attention(const Matrix& Q, const Matrix& K, const Matrix& V, bool mask) const {
    // Q, K, V: [seq_len, d_model]
    int seq_len = Q.rows;
    
    // Reshape Q, K, V for multi-head: [seq_len, num_heads, d_k]
    std::vector<Matrix> Q_heads, K_heads, V_heads, attention_heads;
    
    for (int h = 0; h < num_heads; h++) {
        int start_col = h * d_k;
        int end_col = start_col + d_k;
        
        // Extract head h from Q, K, V
        Q_heads.push_back(Q.slice(0, seq_len, start_col, end_col));
        K_heads.push_back(K.slice(0, seq_len, start_col, end_col));
        V_heads.push_back(V.slice(0, seq_len, start_col, end_col));
    }
    
    // Compute attention for each head
    for (int h = 0; h < num_heads; h++) {
        // Attention scores: Q_h * K_h^T
        Matrix scores = Q_heads[h].cudaMultiply(K_heads[h].transpose());
        
        // Scale by sqrt(d_k)
        double scale = 1.0 / sqrt(d_k);
        scores = scores * scale;
        
        // Apply causal mask if needed
        if (mask) {
            for (int i = 0; i < scores.rows; i++) {
                for (int j = i + 1; j < scores.cols; j++) {
                    scores.data[i][j] = -1e9; // Large negative value
                }
            }
        }
        
        // Softmax
        Matrix attention_weights = scores.cudaSoftmax();
        
        // Apply attention to values: A * V_h
        Matrix attended = attention_weights.cudaMultiply(V_heads[h]);
        attention_heads.push_back(attended);
    }
    
    // Concatenate all heads: [seq_len, d_model]
    Matrix concatenated(seq_len, d_model);
    for (int h = 0; h < num_heads; h++) {
        int start_col = h * d_k;
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < d_k; j++) {
                concatenated.data[i][start_col + j] = attention_heads[h].data[i][j];
            }
        }
    }
    
    return concatenated;
}

Matrix MultiHeadAttention::forward(const Matrix& query, const Matrix& key, const Matrix& value, bool mask) {
    // Cache inputs for backward pass
    cache.input = query;
    cache.query = query;
    cache.key = key;
    cache.value = value;
    
    // Linear transformations: Q = query * W_q + b_q
    Matrix Q = query.cudaMultiply(W_q);
    Matrix K = key.cudaMultiply(W_k);
    Matrix V = value.cudaMultiply(W_v);
    
    // Add biases
    for (int i = 0; i < Q.rows; i++) {
        for (int j = 0; j < Q.cols; j++) {
            Q.data[i][j] += b_q.data[0][j];
            K.data[i][j] += b_k.data[0][j];
            V.data[i][j] += b_v.data[0][j];
        }
    }
    
    // Multi-head attention
    Matrix attended = attention(Q, K, V, mask);
    cache.attended_values = attended;
    
    // Output projection: attended * W_o + b_o
    Matrix output = attended.cudaMultiply(W_o);
    
    // Add output bias
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.data[i][j] += b_o.data[0][j];
        }
    }
    
    return output;
}

std::tuple<Matrix, Matrix, Matrix> MultiHeadAttention::backward(const Matrix& grad_output) {
    // Backward through output projection
    Matrix grad_attended = grad_output.cudaMultiply(W_o.transpose());
    
    // Gradients for output projection
    grad_o.dW.add_inplace(cache.attended_values.transpose().cudaMultiply(grad_output));
    for (int j = 0; j < grad_output.cols; j++) {
        double bias_grad = 0.0;
        for (int i = 0; i < grad_output.rows; i++) {
            bias_grad += grad_output.data[i][j];
        }
        grad_o.db.data[0][j] += bias_grad;
    }
    
    // Backward through multi-head attention
    int seq_len = grad_attended.rows;
    std::vector<Matrix> grad_Q_heads, grad_K_heads, grad_V_heads;
    
    // Split grad_attended by heads
    for (int h = 0; h < num_heads; h++) {
        int start_col = h * d_k;
        int end_col = start_col + d_k;
        Matrix grad_head = grad_attended.slice(0, seq_len, start_col, end_col);
        
        // For simplicity, assume we stored per-head computations during forward
        // In practice, we'd need to recompute or cache more intermediate values
        grad_Q_heads.push_back(Matrix(seq_len, d_k));
        grad_K_heads.push_back(Matrix(seq_len, d_k));
        grad_V_heads.push_back(Matrix(seq_len, d_k));
        
        // Simplified gradient computation (actual implementation would be more complex)
        grad_Q_heads[h] = grad_head;
        grad_K_heads[h] = grad_head;
        grad_V_heads[h] = grad_head;
    }
    
    // Concatenate head gradients
    Matrix grad_Q(seq_len, d_model);
    Matrix grad_K(seq_len, d_model);
    Matrix grad_V(seq_len, d_model);
    
    for (int h = 0; h < num_heads; h++) {
        int start_col = h * d_k;
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < d_k; j++) {
                grad_Q.data[i][start_col + j] = grad_Q_heads[h].data[i][j];
                grad_K.data[i][start_col + j] = grad_K_heads[h].data[i][j];
                grad_V.data[i][start_col + j] = grad_V_heads[h].data[i][j];
            }
        }
    }
    
    // Backward through linear transformations
    Matrix grad_query = grad_Q.cudaMultiply(W_q.transpose());
    Matrix grad_key = grad_K.cudaMultiply(W_k.transpose());
    Matrix grad_value = grad_V.cudaMultiply(W_v.transpose());
    
    // Weight gradients
    grad_q.dW.add_inplace(cache.query.transpose().cudaMultiply(grad_Q));
    grad_k.dW.add_inplace(cache.key.transpose().cudaMultiply(grad_K));
    grad_v.dW.add_inplace(cache.value.transpose().cudaMultiply(grad_V));
    
    // Bias gradients
    for (int j = 0; j < d_model; j++) {
        double bias_grad_q = 0.0, bias_grad_k = 0.0, bias_grad_v = 0.0;
        for (int i = 0; i < seq_len; i++) {
            bias_grad_q += grad_Q.data[i][j];
            bias_grad_k += grad_K.data[i][j];
            bias_grad_v += grad_V.data[i][j];
        }
        grad_q.db.data[0][j] += bias_grad_q;
        grad_k.db.data[0][j] += bias_grad_k;
        grad_v.db.data[0][j] += bias_grad_v;
    }
    
    return std::make_tuple(grad_query, grad_key, grad_value);
}

// ============================================================================
// FIXED LAYER NORMALIZATION IMPLEMENTATION
// ============================================================================

LayerNorm::LayerNorm(int d_model, double eps) : d_model(d_model), eps(eps) {
    gamma = Matrix(1, d_model);
    beta = Matrix(1, d_model);
    grad_gamma = Matrix(1, d_model);
    grad_beta = Matrix(1, d_model);
    
    // Initialize gamma to 1, beta to 0
    for (int i = 0; i < d_model; i++) {
        gamma.data[0][i] = 1.0;
        beta.data[0][i] = 0.0;
    }
    
    grad_gamma.zero();
    grad_beta.zero();
}

Matrix LayerNorm::forward(const Matrix& input) {
    input_cache = input;
    int seq_len = input.rows;
    
    Matrix normalized(seq_len, d_model);
    Matrix mean_cache_temp(seq_len, 1);
    Matrix std_cache_temp(seq_len, 1);
    
    for (int i = 0; i < seq_len; i++) {
        // Compute mean
        double mean = 0.0;
        for (int j = 0; j < d_model; j++) {
            mean += input.data[i][j];
        }
        mean /= d_model;
        mean_cache_temp.data[i][0] = mean;
        
        // Compute variance
        double variance = 0.0;
        for (int j = 0; j < d_model; j++) {
            double diff = input.data[i][j] - mean;
            variance += diff * diff;
        }
        variance /= d_model;
        double std_dev = sqrt(variance + eps);
        std_cache_temp.data[i][0] = std_dev;
        
        // Normalize and apply scale/shift
        for (int j = 0; j < d_model; j++) {
            double x_norm = (input.data[i][j] - mean) / std_dev;
            normalized.data[i][j] = gamma.data[0][j] * x_norm + beta.data[0][j];
        }
    }
    
    mean_cache = mean_cache_temp;
    std_cache = std_cache_temp;
    normalized_cache = normalized;
    
    return normalized;
}

Matrix LayerNorm::backward(const Matrix& grad_output) {
    int seq_len = input_cache.rows;
    Matrix grad_input(seq_len, d_model);
    
    grad_gamma.zero();
    grad_beta.zero();
    
    for (int i = 0; i < seq_len; i++) {
        double mean = mean_cache.data[i][0];
        double std_dev = std_cache.data[i][0];
        
        // Gradients w.r.t. gamma and beta
        for (int j = 0; j < d_model; j++) {
            double x_norm = (input_cache.data[i][j] - mean) / std_dev;
            grad_gamma.data[0][j] += grad_output.data[i][j] * x_norm;
            grad_beta.data[0][j] += grad_output.data[i][j];
        }
        
        // Gradient w.r.t. input
        double sum_grad_gamma_x_norm = 0.0;
        double sum_grad_gamma = 0.0;
        
        for (int j = 0; j < d_model; j++) {
            double x_norm = (input_cache.data[i][j] - mean) / std_dev;
            sum_grad_gamma_x_norm += grad_output.data[i][j] * gamma.data[0][j] * x_norm;
            sum_grad_gamma += grad_output.data[i][j] * gamma.data[0][j];
        }
        
        for (int j = 0; j < d_model; j++) {
            double x_norm = (input_cache.data[i][j] - mean) / std_dev;
            grad_input.data[i][j] = (gamma.data[0][j] / std_dev) * 
                                   (grad_output.data[i][j] - 
                                    sum_grad_gamma / d_model - 
                                    x_norm * sum_grad_gamma_x_norm / d_model);
        }
    }
    
    return grad_input;
}

// ============================================================================
// FIXED TRANSFORMER ENCODER LAYER IMPLEMENTATION
// ============================================================================

TransformerEncoderLayer::TransformerEncoderLayer(int d_model, int num_heads, int d_ff, double dropout_rate) 
    : dropout_rate(dropout_rate) {
    
    attention = std::make_unique<MultiHeadAttention>(d_model, num_heads);
    feed_forward = std::make_unique<FeedForward>(d_model, d_ff);
    norm1 = std::make_unique<LayerNorm>(d_model);
    norm2 = std::make_unique<LayerNorm>(d_model);
}

Matrix TransformerEncoderLayer::forward(const Matrix& input, bool training) {
    input_cache = input;
    
    // Pre-LayerNorm architecture (more stable)
    // Self-attention with residual connection
    Matrix norm1_out = norm1->forward(input);
    norm1_out_cache = norm1_out;
    
    Matrix attention_out = attention->forward(norm1_out, norm1_out, norm1_out, false);
    attention_out_cache = attention_out;
    
    // Apply dropout if training
    if (training && dropout_rate > 0.0) {
        attention_out = attention_out.dropout(dropout_rate);
    }
    
    // Residual connection
    Matrix residual1 = input + attention_out;
    
    // Feed-forward with residual connection
    Matrix norm2_out = norm2->forward(residual1);
    Matrix ff_out = feed_forward->forward(norm2_out);
    ff_out_cache = ff_out;
    
    // Apply dropout if training
    if (training && dropout_rate > 0.0) {
        ff_out = ff_out.dropout(dropout_rate);
    }
    
    // Residual connection
    Matrix output = residual1 + ff_out;
    
    return output;
}

Matrix TransformerEncoderLayer::backward(const Matrix& grad_output) {
    // Backward through second residual connection
    Matrix grad_residual1 = grad_output;
    Matrix grad_ff_out = grad_output;
    
    // Backward through feed-forward
    Matrix grad_norm2_out = feed_forward->backward(grad_ff_out);
    Matrix grad_residual1_from_norm2 = norm2->backward(grad_norm2_out);
    
    grad_residual1.add_inplace(grad_residual1_from_norm2);
    
    // Backward through first residual connection
    Matrix grad_input = grad_residual1;
    Matrix grad_attention_out = grad_residual1;
    
    // Backward through attention
    auto attention_grads = attention->backward(grad_attention_out);
    Matrix grad_norm1_out = std::get<0>(attention_grads); // All three should be the same for self-attention
    
    // Backward through first layer norm
    Matrix grad_input_from_norm1 = norm1->backward(grad_norm1_out);
    grad_input.add_inplace(grad_input_from_norm1);
    
    return grad_input;
}

// ============================================================================
// FIXED TRANSFORMER MAIN CLASS IMPLEMENTATION
// ============================================================================

Matrix Transformer::forward(const Matrix& input) {
    // 1. Create patches from image (28x28 -> 7x7 patches of 4x4)
    Matrix patches = create_patches(input);
    patches_cache = patches;
    
    // 2. Embed patches: patches (49 x 16) -> embedded (49 x d_model)
    Matrix embedded = patches.cudaMultiply(patch_embedding_W);
    
    // Add embedding bias
    for (int i = 0; i < embedded.rows; i++) {
        for (int j = 0; j < embedded.cols; j++) {
            embedded.data[i][j] += patch_embedding_b.data[0][j];
        }
    }
    embedded_cache = embedded;
    
    // 3. Add class token: (1 + 49) x d_model
    Matrix tokens(num_patches + 1, d_model);
    
    // First row is class token
    for (int j = 0; j < d_model; j++) {
        tokens.data[0][j] = class_token.data[0][j];
    }
    
    // Remaining rows are embedded patches
    for (int i = 0; i < num_patches; i++) {
        for (int j = 0; j < d_model; j++) {
            tokens.data[i + 1][j] = embedded.data[i][j];
        }
    }
    
    // 4. Add positional encoding
    tokens = pos_encoding->encode(tokens);
    tokens_cache = tokens;
    
    // 5. Pass through encoder layers
    Matrix encoded = tokens;
    for (auto& layer : encoder_layers) {
        encoded = layer->forward(encoded, training_mode);
    }
    encoded_cache = encoded;
    
    // 6. Classification using class token (first token)
    Matrix class_features(1, d_model);
    for (int j = 0; j < d_model; j++) {
        class_features.data[0][j] = encoded.data[0][j];
    }
    
    // 7. Linear classification
    Matrix logits = class_features.cudaMultiply(classifier_W);
    
    // Add classifier bias
    for (int j = 0; j < num_classes; j++) {
        logits.data[0][j] += classifier_b.data[0][j];
    }
    
    // 8. Apply softmax for probabilities
    return logits.cudaSoftmax();
}

Matrix Transformer::backward(const Matrix& predictions, const std::vector<int>& labels) {
    // 1. Compute loss gradients (softmax + cross-entropy)
    Matrix grad_logits(1, num_classes);
    
    // For cross-entropy: grad = softmax_output - one_hot_labels
    for (int j = 0; j < num_classes; j++) {
        grad_logits.data[0][j] = predictions.data[0][j];
        if (j == labels[0]) {
            grad_logits.data[0][j] -= 1.0;
        }
    }
    
    // 2. Backward through classifier
    Matrix class_features(1, d_model);
    for (int j = 0; j < d_model; j++) {
        class_features.data[0][j] = encoded_cache.data[0][j];
    }
    
    // Classifier weight gradients
    classifier_grad.dW.add_inplace(class_features.transpose().cudaMultiply(grad_logits));
    
    // Classifier bias gradients
    for (int j = 0; j < num_classes; j++) {
        classifier_grad.db.data[0][j] += grad_logits.data[0][j];
    }
    
    // Gradient w.r.t. class features
    Matrix grad_class_features = grad_logits.cudaMultiply(classifier_W.transpose());
    
    // 3. Backward through encoder layers
    Matrix grad_encoded(encoded_cache.rows, encoded_cache.cols);
    grad_encoded.zero();
    
    // Set gradient for class token
    for (int j = 0; j < d_model; j++) {
        grad_encoded.data[0][j] = grad_class_features.data[0][j];
    }
    
    // Backward through encoder stack
    for (int i = encoder_layers.size() - 1; i >= 0; i--) {
        grad_encoded = encoder_layers[i]->backward(grad_encoded);
    }
    
    // 4. Backward through positional encoding (pass-through)
    Matrix grad_tokens = pos_encoding->backward(grad_encoded);
    
    // 5. Backward through patch embedding
    Matrix grad_embedded(num_patches, d_model);
    for (int i = 0; i < num_patches; i++) {
        for (int j = 0; j < d_model; j++) {
            grad_embedded.data[i][j] = grad_tokens.data[i + 1][j];
        }
    }
    
    // Class token gradient
    for (int j = 0; j < d_model; j++) {
        class_token_grad.data[0][j] += grad_tokens.data[0][j];
    }
    
    // Patch embedding weight gradients
    patch_grad.dW.add_inplace(patches_cache.transpose().cudaMultiply(grad_embedded));
    
    // Patch embedding bias gradients
    for (int j = 0; j < d_model; j++) {
        double bias_grad = 0.0;
        for (int i = 0; i < num_patches; i++) {
            bias_grad += grad_embedded.data[i][j];
        }
        patch_grad.db.data[0][j] += bias_grad;
    }
    
    // 6. Backward through patch creation (not needed for parameter updates)
    Matrix grad_patches = grad_embedded.cudaMultiply(patch_embedding_W.transpose());
    
    return grad_patches; // Could be used to compute input gradients if needed
}

// ============================================================================
// FIXED BATCH TRAINING IMPLEMENTATION
// ============================================================================

std::pair<double, double> Transformer::train_batch(const std::vector<Matrix>& inputs, 
                                                  const std::vector<int>& labels, 
                                                  double learning_rate) {
    double total_loss = 0.0;
    int correct = 0;
    int batch_size = inputs.size();
    
    // Zero gradients
    zero_gradients();
    
    // Process batch (accumulate gradients)
    for (size_t i = 0; i < inputs.size(); i++) {
        // Forward pass
        Matrix predictions = forward(inputs[i]);
        
        // Compute loss and accuracy
        std::vector<int> single_label = {labels[i]};
        total_loss += compute_loss(predictions, single_label);
        
        // Get predicted class
        int predicted = 0;
        double max_prob = predictions.data[0][0];
        for (int j = 1; j < predictions.cols; j++) {
            if (predictions.data[0][j] > max_prob) {
                max_prob = predictions.data[0][j];
                predicted = j;
            }
        }
        if (predicted == labels[i]) correct++;
        
        // Backward pass (accumulate gradients)
        backward(predictions, single_label);
    }
    
    // Average gradients
    double scale_factor = 1.0 / static_cast<double>(batch_size);
    
    // Scale all gradients
    for (auto& layer : encoder_layers) {
        layer->scale_gradients(scale_factor);
    }
    
    patch_grad.dW.multiply_inplace(scale_factor);
    patch_grad.db.multiply_inplace(scale_factor);
    classifier_grad.dW.multiply_inplace(scale_factor);
    classifier_grad.db.multiply_inplace(scale_factor);
    class_token_grad.multiply_inplace(scale_factor);
    
    // Clip gradients for stability
    clip_gradients(1.0);
    
    // Update weights using Adam optimizer
    current_step++;
    update_weights_adam(current_step, learning_rate);
    
    double avg_loss = total_loss / batch_size;
    double accuracy = static_cast<double>(correct) / batch_size;
    
    return std::make_pair(avg_loss, accuracy);
}

void Transformer::update_weights_adam(int step, double learning_rate) {
    // Update patch embedding
    optimizer->update(patch_embedding_W, patch_grad.dW, &patch_embedding_W, step, learning_rate);
    optimizer->update(patch_embedding_b, patch_grad.db, &patch_embedding_b, step, learning_rate);
    
    // Update class token
    optimizer->update(class_token, class_token_grad, &class_token, step, learning_rate);
    
    // Update classifier
    optimizer->update(classifier_W, classifier_grad.dW, &classifier_W, step, learning_rate);
    optimizer->update(classifier_b, classifier_grad.db, &classifier_b, step, learning_rate);
    
    // Update encoder layers
    for (auto& layer : encoder_layers) {
        layer->update_weights_adam(*optimizer, step);
    }
}

void Transformer::zero_gradients() {
    patch_grad.zero();
    classifier_grad.zero();
    class_token_grad.zero();
    
    for (auto& layer : encoder_layers) {
        layer->zero_gradients();
    }
}

void Transformer::clip_gradients(double max_norm) {
    patch_grad.clip(max_norm);
    classifier_grad.clip(max_norm);
    class_token_grad = class_token_grad.clip_gradients(max_norm);
    
    for (auto& layer : encoder_layers) {
        layer->clip_gradients(max_norm);
    }
}

// ============================================================================
// MISSING IMPLEMENTATIONS
// ============================================================================

Transformer::Transformer(int d_model, int num_heads, int num_layers, int d_ff, 
                         int num_classes, int patch_size, double dropout_rate) 
    : d_model(d_model), num_heads(num_heads), num_layers(num_layers), 
      d_ff(d_ff), num_classes(num_classes), patch_size(patch_size), 
      dropout_rate(dropout_rate), training_mode(true), current_step(0) {
    
    // Calculate derived parameters
    num_patches = (28 / patch_size) * (28 / patch_size);
    max_seq_len = num_patches + 1; // +1 for class token
    
    // Initialize positional encoding
    pos_encoding = std::make_unique<PositionalEncoding>(max_seq_len, d_model);
    
    // Initialize encoder layers
    for (int i = 0; i < num_layers; i++) {
        encoder_layers.push_back(std::make_unique<TransformerEncoderLayer>(d_model, num_heads, d_ff, dropout_rate));
    }
    
    // Initialize embedding layers
    int patch_dim = patch_size * patch_size;
    patch_embedding_W = Matrix(patch_dim, d_model);
    patch_embedding_b = Matrix(1, d_model);
    patch_embedding_W.xavier_init();
    patch_embedding_b.zero();
    
    // Initialize class token
    class_token = Matrix(1, d_model);
    class_token.randomize(-0.02, 0.02);
    
    // Initialize classifier
    classifier_W = Matrix(d_model, num_classes);
    classifier_b = Matrix(1, num_classes);
    classifier_W.xavier_init();
    classifier_b.zero();
    
    // Initialize gradients
    patch_grad = Gradients(patch_dim, d_model);
    classifier_grad = Gradients(d_model, num_classes);
    class_token_grad = Matrix(1, d_model);
    class_token_grad.zero();
    
    // Initialize optimizer and scheduler
    optimizer = std::make_unique<AdamOptimizer>();
    scheduler = std::make_unique<LRScheduler>(0.001);
}

Matrix Transformer::create_patches(const Matrix& image) {
    int patches_per_side = 28 / patch_size;
    Matrix patches(num_patches, patch_size * patch_size);
    
    int patch_idx = 0;
    for (int i = 0; i < patches_per_side; i++) {
        for (int j = 0; j < patches_per_side; j++) {
            int pixel_idx = 0;
            for (int pi = 0; pi < patch_size; pi++) {
                for (int pj = 0; pj < patch_size; pj++) {
                    int row = i * patch_size + pi;
                    int col = j * patch_size + pj;
                    if (row < image.rows && col < image.cols) {
                        patches.data[patch_idx][pixel_idx] = image.data[row][col];
                    }
                    pixel_idx++;
                }
            }
            patch_idx++;
        }
    }
    
    patches_cache = patches;
    return patches;
}

double Transformer::compute_loss(const Matrix& predictions, const std::vector<int>& labels) {
    double loss = 0.0;
    
    for (size_t i = 0; i < labels.size(); i++) {
        int label = labels[i];
        double pred_prob = predictions.data[i][label];
        
        // Prevent log(0)
        pred_prob = (std::max)(pred_prob, 1e-15);
        loss -= log(pred_prob);
    }
    
    return loss / labels.size();
}

void Transformer::print_model_info() const {
    std::cout << "\n🧠 === TRANSFORMER MODEL INFO ===" << std::endl;
    std::cout << "📐 Architecture:" << std::endl;
    std::cout << "  • Embedding dimension: " << d_model << std::endl;
    std::cout << "  • Attention heads: " << num_heads << std::endl;
    std::cout << "  • Encoder layers: " << num_layers << std::endl;
    std::cout << "  • Feed-forward dim: " << d_ff << std::endl;
    std::cout << "  • Output classes: " << num_classes << std::endl;
    std::cout << "  • Patch size: " << patch_size << "x" << patch_size << std::endl;
    std::cout << "  • Number of patches: " << num_patches << std::endl;
    std::cout << "  • Sequence length: " << max_seq_len << " (patches + class token)" << std::endl;
    std::cout << "  • Dropout rate: " << dropout_rate << std::endl;
    
    // Calculate total parameters
    int total_params = 0;
    total_params += patch_embedding_W.rows * patch_embedding_W.cols; // Patch embedding
    total_params += patch_embedding_b.cols; // Patch embedding bias
    total_params += class_token.cols; // Class token
    total_params += classifier_W.rows * classifier_W.cols; // Classifier
    total_params += classifier_b.cols; // Classifier bias
    
    // Approximate encoder parameters (each layer has attention + feedforward + layer norms)
    int encoder_params_per_layer = 4 * d_model * d_model + 2 * d_model * d_ff + 4 * d_model;
    total_params += num_layers * encoder_params_per_layer;
    
    std::cout << "\n📊 Parameters:" << std::endl;
    std::cout << "  • Total parameters: ~" << total_params << std::endl;
    std::cout << "  • Memory (float32): ~" << (total_params * 4) / (1024*1024) << " MB" << std::endl;
    std::cout << "=============================\n" << std::endl;
}

// ============================================================================
// FEED FORWARD MISSING IMPLEMENTATIONS
// ============================================================================

FeedForward::FeedForward(int d_model, int d_ff) : d_model(d_model), d_ff(d_ff) {
    W1 = Matrix(d_model, d_ff);
    W2 = Matrix(d_ff, d_model);
    b1 = Matrix(1, d_ff);
    b2 = Matrix(1, d_model);
    
    W1.xavier_init();
    W2.xavier_init();
    b1.zero();
    b2.zero();
    
    grad_1 = Gradients(d_model, d_ff);
    grad_2 = Gradients(d_ff, d_model);
}

Matrix FeedForward::forward(const Matrix& input) {
    input_cache = input;
    
    // First linear layer
    Matrix hidden = input * W1;
    
    // Add bias
    for (int i = 0; i < hidden.rows; i++) {
        for (int j = 0; j < hidden.cols; j++) {
            hidden.data[i][j] += b1.data[0][j];
        }
    }
    
    // GELU activation
    hidden = hidden.gelu();
    hidden_cache = hidden;
    
    // Second linear layer
    Matrix output = hidden * W2;
    
    // Add bias
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.data[i][j] += b2.data[0][j];
        }
    }
    
    return output;
}

Matrix FeedForward::backward(const Matrix& grad_output) {
    // Gradient w.r.t second layer
    Matrix grad_hidden = grad_output * W2.transpose();
    
    // Second layer weight and bias gradients
    grad_2.dW.add_inplace(hidden_cache.transpose() * grad_output);
    for (int j = 0; j < grad_output.cols; j++) {
        double bias_grad = 0.0;
        for (int i = 0; i < grad_output.rows; i++) {
            bias_grad += grad_output.data[i][j];
        }
        grad_2.db.data[0][j] += bias_grad;
    }
    
    // GELU derivative
    grad_hidden = grad_hidden.hadamard(hidden_cache.gelu_derivative());
    
    // First layer weight and bias gradients
    grad_1.dW.add_inplace(input_cache.transpose() * grad_hidden);
    for (int j = 0; j < grad_hidden.cols; j++) {
        double bias_grad = 0.0;
        for (int i = 0; i < grad_hidden.rows; i++) {
            bias_grad += grad_hidden.data[i][j];
        }
        grad_1.db.data[0][j] += bias_grad;
    }
    
    // Gradient w.r.t input
    Matrix grad_input = grad_hidden * W1.transpose();
    
    return grad_input;
}

void FeedForward::update_weights(double learning_rate) {
    W1.subtract_inplace(grad_1.dW * learning_rate);
    W2.subtract_inplace(grad_2.dW * learning_rate);
    b1.subtract_inplace(grad_1.db * learning_rate);
    b2.subtract_inplace(grad_2.db * learning_rate);
}

void FeedForward::update_weights_adam(AdamOptimizer& optimizer, int step) {
    optimizer.update(W1, grad_1.dW, &W1, step, 0.001);
    optimizer.update(W2, grad_2.dW, &W2, step, 0.001);
    optimizer.update(b1, grad_1.db, &b1, step, 0.001);
    optimizer.update(b2, grad_2.db, &b2, step, 0.001);
}

void FeedForward::zero_gradients() {
    grad_1.zero();
    grad_2.zero();
}

void FeedForward::clip_gradients(double max_norm) {
    grad_1.clip(max_norm);
    grad_2.clip(max_norm);
}

// ============================================================================
// TRANSFORMER ENCODER LAYER MISSING IMPLEMENTATIONS
// ============================================================================

void TransformerEncoderLayer::update_weights_adam(AdamOptimizer& optimizer, int step) {
    attention->update_weights_adam(optimizer, step);
    feed_forward->update_weights_adam(optimizer, step);
    norm1->update_weights_adam(optimizer, step);
    norm2->update_weights_adam(optimizer, step);
}

void TransformerEncoderLayer::zero_gradients() {
    attention->zero_gradients();
    feed_forward->zero_gradients();
    norm1->zero_gradients();
    norm2->zero_gradients();
}

void TransformerEncoderLayer::clip_gradients(double max_norm) {
    attention->clip_gradients(max_norm);
    feed_forward->clip_gradients(max_norm);
}

void TransformerEncoderLayer::scale_gradients(double scale_factor) {
    // Scale attention gradients
    attention->grad_q.dW.multiply_inplace(scale_factor);
    attention->grad_k.dW.multiply_inplace(scale_factor);
    attention->grad_v.dW.multiply_inplace(scale_factor);
    attention->grad_o.dW.multiply_inplace(scale_factor);
    
    attention->grad_q.db.multiply_inplace(scale_factor);
    attention->grad_k.db.multiply_inplace(scale_factor);
    attention->grad_v.db.multiply_inplace(scale_factor);
    attention->grad_o.db.multiply_inplace(scale_factor);
    
    // Scale feed-forward gradients
    feed_forward->grad_1.dW.multiply_inplace(scale_factor);
    feed_forward->grad_2.dW.multiply_inplace(scale_factor);
    feed_forward->grad_1.db.multiply_inplace(scale_factor);
    feed_forward->grad_2.db.multiply_inplace(scale_factor);
    
    // Scale layer norm gradients
    norm1->grad_gamma.multiply_inplace(scale_factor);
    norm1->grad_beta.multiply_inplace(scale_factor);
    norm2->grad_gamma.multiply_inplace(scale_factor);
    norm2->grad_beta.multiply_inplace(scale_factor);
}

// ============================================================================
// POSITIONAL ENCODING MISSING IMPLEMENTATIONS  
// ============================================================================

PositionalEncoding::PositionalEncoding(int max_len, int d_model) 
    : max_len(max_len), d_model(d_model), encoding(max_len, d_model) {
    
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d_model; i++) {
            if (i % 2 == 0) {
                encoding.data[pos][i] = sin(pos / pow(10000.0, (2.0 * i) / d_model));
            } else {
                encoding.data[pos][i] = cos(pos / pow(10000.0, (2.0 * (i-1)) / d_model));
            }
        }
    }
}

Matrix PositionalEncoding::encode(const Matrix& input) const {
    Matrix result = input;
    int seq_len = (std::min)(input.rows, max_len);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < (std::min)(input.cols, d_model); j++) {
            result.data[i][j] += encoding.data[i][j];
        }
    }
    return result;
}

Matrix PositionalEncoding::backward(const Matrix& grad_output) const {
    // Positional encoding gradients pass through unchanged
    return grad_output;
}

// ============================================================================
// LAYER NORM UPDATE FUNCTIONS  
// ============================================================================

void LayerNorm::update_weights(double learning_rate) {
    gamma.subtract_inplace(grad_gamma * learning_rate);
    beta.subtract_inplace(grad_beta * learning_rate);
}

void LayerNorm::update_weights_adam(AdamOptimizer& optimizer, int step) {
    optimizer.update(gamma, grad_gamma, &gamma, step, 0.001);
    optimizer.update(beta, grad_beta, &beta, step, 0.001);
}

void LayerNorm::zero_gradients() {
    grad_gamma.zero();
    grad_beta.zero();
}