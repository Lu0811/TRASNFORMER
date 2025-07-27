// PARCHE PARA transformer.cpp - Añadir estas funciones de mejor inicialización
// Puedes copiar y pegar estas mejoras en tu transformer.cpp existente

// AÑADIR AL INICIO DE transformer.cpp (después de los includes)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// REEMPLAZAR el constructor de MultiHeadAttention con este mejorado:
MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads) 
    : d_model(d_model), num_heads(num_heads), head_dim(d_model / num_heads) {
    
    // Inicialización mejorada con escala apropiada
    double scale = sqrt(2.0 / d_model);  // He initialization
    
    W_q = Matrix(d_model, d_model);
    W_k = Matrix(d_model, d_model);
    W_v = Matrix(d_model, d_model);
    W_o = Matrix(d_model, d_model);
    
    // Inicialización más conservadora para mejor accuracy inicial
    W_q.xavier_init(scale * 0.5);  // Queries más conservadoras
    W_k.xavier_init(scale * 0.5);  // Keys más conservadoras
    W_v.xavier_init(scale * 0.5);  // Values más conservadoras
    W_o.xavier_init(scale * 0.8);  // Output projection menos agresiva
}

// REEMPLAZAR el constructor de FeedForward con este mejorado:
FeedForward::FeedForward(int d_model, int d_ff) : d_model(d_model), d_ff(d_ff) {
    W1 = Matrix(d_model, d_ff);
    b1 = Matrix(1, d_ff);
    W2 = Matrix(d_ff, d_model);
    b2 = Matrix(1, d_model);
    
    // Inicialización mejorada para convergencia rápida
    double scale1 = sqrt(2.0 / d_model);    // He init para primera capa
    double scale2 = sqrt(1.0 / d_ff) * 0.5; // Más conservador para segunda capa
    
    W1.xavier_init(scale1);
    W2.xavier_init(scale2);
    
    // Bias inicializado a valores pequeños pero no cero
    for (int i = 0; i < b1.cols; i++) {
        b1.data[0][i] = 0.01;  // Pequeño bias positivo para GELU
    }
    for (int i = 0; i < b2.cols; i++) {
        b2.data[0][i] = 0.0;   // Output bias cero
    }
}

// MODIFICAR el constructor de Transformer para mejor inicialización:
// En la parte de classification_head, reemplazar con:

    // Classification head con inicialización muy cuidadosa
    classification_head = Matrix(d_model, num_classes);
    double class_scale = sqrt(1.0 / d_model) * 0.1;  // Muy conservador
    classification_head.xavier_init(class_scale);
    
    // AÑADIR: Bias para clasificación inicializado para probabilidades uniformes
    classification_bias = Matrix(1, num_classes);
    double uniform_logit = log(1.0 / num_classes);  // log(0.1) para 10 clases
    for (int i = 0; i < num_classes; i++) {
        classification_bias.data[0][i] = uniform_logit;
    }

// MODIFICAR la función forward() de Transformer para usar el bias:
// En la parte final donde aplica classification_head, cambiar a:

    // Classification con bias
    Matrix logits = pooled.multiply(classification_head);
    // AÑADIR el bias
    for (int i = 0; i < logits.cols; i++) {
        logits.data[0][i] += classification_bias.data[0][i];
    }
    
    return logits.softmax();

// AÑADIR esta nueva función para learning rate con warmup:
double get_learning_rate_with_warmup(int step, int warmup_steps, double base_lr, double min_lr = 0.0001) {
    if (step < warmup_steps) {
        // Warmup lineal desde min_lr hasta base_lr
        double warmup_progress = (double)step / warmup_steps;
        return min_lr + (base_lr - min_lr) * warmup_progress;
    } else {
        // Después del warmup, usar cosine decay
        int decay_steps = step - warmup_steps;
        double max_decay_steps = 1000.0;  // Ajustar según necesidad
        double decay_progress = std::min((double)decay_steps / max_decay_steps, 1.0);
        double decay_factor = 0.5 * (1 + cos(M_PI * decay_progress));
        return min_lr + (base_lr - min_lr) * decay_factor;
    }
}

// MODIFICAR la función train_batch para usar warmup:
// Al inicio de train_batch, añadir:

std::pair<double, double> Transformer::train_batch(
    const std::vector<Matrix>& batch_images,
    const std::vector<int>& batch_labels,
    double base_learning_rate) {
    
    // Contador estático para tracking global de steps
    static int global_step = 0;
    const int warmup_steps = 50;  // Ajustable
    
    // Calcular learning rate con warmup
    double learning_rate = get_learning_rate_with_warmup(global_step, warmup_steps, base_learning_rate);
    global_step++;
    
    // Si es uno de los primeros batches, imprimir el LR actual
    if (global_step <= 10 || global_step == warmup_steps) {
        std::cout << "[Step " << global_step << "] Learning rate: " << learning_rate << std::endl;
    }
    
    // ... resto del código de train_batch usando 'learning_rate' en lugar de 'base_learning_rate' ...

// MODIFICAR el positional encoding en el constructor de Transformer:
// Hacer el positional encoding más suave:

    // Positional encoding mejorado (escalado)
    positional_encoding = Matrix(num_patches, d_model);
    double pe_scale = 0.1;  // Escalar para no dominar los embeddings
    
    for (int pos = 0; pos < num_patches; pos++) {
        for (int i = 0; i < d_model; i++) {
            if (i % 2 == 0) {
                positional_encoding.data[pos][i] = sin(pos / pow(10000.0, i / double(d_model))) * pe_scale;
            } else {
                positional_encoding.data[pos][i] = cos(pos / pow(10000.0, (i-1) / double(d_model))) * pe_scale;
            }
        }
    }

// NOTA: No olvides declarar classification_bias en transformer.h:
// En la clase Transformer, añadir:
// Matrix classification_bias;