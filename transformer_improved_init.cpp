// Fragmento mejorado para transformer.cpp - Mejor inicialización

// En el constructor de cada capa, cambiar la inicialización:

// MEJOR INICIALIZACIÓN PARA MULTI-HEAD ATTENTION
MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads) 
    : d_model(d_model), num_heads(num_heads), head_dim(d_model / num_heads) {
    
    // Inicialización mejorada con escala apropiada
    double scale = sqrt(2.0 / d_model);  // He initialization mejorado
    
    W_q = Matrix(d_model, d_model);
    W_k = Matrix(d_model, d_model);
    W_v = Matrix(d_model, d_model);
    W_o = Matrix(d_model, d_model);
    
    // Inicialización más conservadora para mejor accuracy inicial
    W_q.xavier_init(scale * 0.5);  // Escala reducida
    W_k.xavier_init(scale * 0.5);
    W_v.xavier_init(scale * 0.5);
    W_o.xavier_init(scale * 0.8);  // Output más estable
}

// MEJOR INICIALIZACIÓN PARA FEED-FORWARD
FeedForward::FeedForward(int d_model, int d_ff) : d_model(d_model), d_ff(d_ff) {
    W1 = Matrix(d_model, d_ff);
    b1 = Matrix(1, d_ff);
    W2 = Matrix(d_ff, d_model);
    b2 = Matrix(1, d_model);
    
    // Inicialización mejorada para convergencia rápida
    double scale1 = sqrt(2.0 / d_model);
    double scale2 = sqrt(1.0 / d_ff);  // Más conservador para la segunda capa
    
    W1.xavier_init(scale1);
    W2.xavier_init(scale2 * 0.5);  // Salida más estable
    
    // Bias inicializado a valores pequeños pero no cero
    for (int i = 0; i < b1.cols; i++) {
        b1.data[0][i] = 0.01;  // Pequeño bias positivo
    }
    for (int i = 0; i < b2.cols; i++) {
        b2.data[0][i] = 0.0;
    }
}

// MEJOR INICIALIZACIÓN PARA TRANSFORMER
Transformer::Transformer(int d_model, int num_heads, int num_layers, int d_ff, 
                       int num_classes, int patch_size, double dropout_rate)
    : d_model(d_model), num_heads(num_heads), num_layers(num_layers), 
      num_classes(num_classes), patch_size(patch_size), dropout_rate(dropout_rate),
      training(true) {
    
    // Patch embedding con mejor inicialización
    int num_patches = (28 / patch_size) * (28 / patch_size);
    patch_embedding = Matrix(patch_size * patch_size, d_model);
    
    // Inicialización más cuidadosa para embeddings
    double embed_scale = sqrt(1.0 / (patch_size * patch_size));
    patch_embedding.xavier_init(embed_scale);
    
    // Positional encoding mejorado
    positional_encoding = Matrix(num_patches, d_model);
    for (int pos = 0; pos < num_patches; pos++) {
        for (int i = 0; i < d_model; i++) {
            if (i % 2 == 0) {
                positional_encoding.data[pos][i] = sin(pos / pow(10000.0, i / double(d_model))) * 0.1;
            } else {
                positional_encoding.data[pos][i] = cos(pos / pow(10000.0, (i-1) / double(d_model))) * 0.1;
            }
        }
    }
    
    // Classification head con inicialización cuidadosa
    classification_head = Matrix(d_model, num_classes);
    classification_head.xavier_init(sqrt(1.0 / d_model) * 0.1);  // Muy conservador
    
    // Añadir bias para clasificación
    classification_bias = Matrix(1, num_classes);
    // Inicializar bias con log(1/num_classes) para probabilidades uniformes
    double uniform_logit = log(1.0 / num_classes);
    for (int i = 0; i < num_classes; i++) {
        classification_bias.data[0][i] = uniform_logit;
    }
}

// FUNCIÓN DE LEARNING RATE CON WARMUP
double get_learning_rate_with_warmup(int step, int warmup_steps, double base_lr) {
    if (step < warmup_steps) {
        // Warmup lineal desde 0 hasta base_lr
        return base_lr * (double(step) / warmup_steps);
    } else {
        // Después del warmup, usar cosine decay
        int decay_steps = step - warmup_steps;
        double decay_factor = 0.5 * (1 + cos(M_PI * decay_steps / 1000.0));
        return base_lr * decay_factor;
    }
}

// MODIFICACIÓN EN train_batch PARA USAR WARMUP
std::pair<double, double> Transformer::train_batch(
    const std::vector<Matrix>& batch_images,
    const std::vector<int>& batch_labels,
    double base_learning_rate) {
    
    static int global_step = 0;  // Contador global de steps
    const int warmup_steps = 100;  // Primeros 100 batches con warmup
    
    // Learning rate con warmup
    double learning_rate = get_learning_rate_with_warmup(global_step, warmup_steps, base_learning_rate);
    global_step++;
    
    // Resto del código de entrenamiento...
    
    // También añadir gradient clipping para estabilidad
    const double max_grad_norm = 1.0;
    // Aplicar clipping a los gradientes antes de actualizar pesos
}