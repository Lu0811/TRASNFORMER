#include <iostream> 
#include <cuda_runtime.h> 
int main() { 
    printf("⚡ VERIFICACIÓN MODO BALANCEADO\\n\\n"); 
    if(count > 0) { 
        printf("GPU: %s ^(%zu MB^)\\n", prop.name, prop.totalGlobalMem/1024/1024); 
        printf("\\n🎯 CONFIGURACIÓN BALANCEADA:\\n"); 
        printf("- Modelo: 2M params ^(balanceado^)\\n"); 
        printf("- Batch Size: 48\\n"); 
        printf("- Learning Rate: 0.0008 con warmup\\n"); 
        printf("- Dataset: 40K muestras\\n"); 
        printf("- Patches: 49 ^(4x4^)\\n"); 
        printf("\\n✅ LISTO PARA VELOCIDAD + ACCURACY\\n"); 
    } 
    return 0; 
} 
