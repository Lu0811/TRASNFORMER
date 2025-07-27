#include <iostream> 
#include <cuda_runtime.h> 
int main() { 
    printf("⚡ VERIFICACIÓN ULTRA-SIMPLE\\n"); 
    for(int i=0; i<count; i++) { 
        printf("GPU %d: %s ^(%zu MB^)\\n", i, prop.name, prop.totalGlobalMem/1024/1024); 
        if(prop.totalGlobalMem > 3000000000) { 
            printf("  ⚡ DEDICADA - LISTA PARA SATURACIÓN\\n"); 
        } 
    } 
    printf("\\n🎯 CONFIGURADO PARA ^>80% GPU\\n"); 
    return 0; 
} 
