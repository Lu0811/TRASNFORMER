#include <iostream> 
#include <cuda_runtime.h> 
#include <cublas_v2.h> 
int main() { 
    printf("🔥 VERIFICACIÓN MODO INTENSIVO\\n"); 
    printf("GPUs detectadas: %d\\n", count); 
    for(int i=0; i<count; i++) { 
        printf("GPU %d: %s\\n", i, prop.name); 
        printf("  Memoria: %zu MB\\n", prop.totalGlobalMem/1024/1024); 
        printf("  Multiprocessors: %d\\n", prop.multiProcessorCount); 
        printf("  Max threads/block: %d\\n", prop.maxThreadsPerBlock); 
        if(prop.totalGlobalMem > 2000000000) { 
            printf("  🔥 GPU DEDICADA - LISTA PARA SATURACIÓN\\n"); 
        } else { 
            printf("  ⚠️  GPU integrada - saturación limitada\\n"); 
        } 
    } 
    size_t free_mem, total_mem; 
    printf("\\n💾 Memoria GPU disponible: %zu MB\\n", free_mem/1024/1024); 
    printf("🎯 CONFIGURADO PARA USAR ^>80% GPU\\n"); 
    return 0; 
} 
