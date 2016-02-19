#include <stdio.h>

void cpuid(int fn, unsigned int *eax, unsigned int *ebx, 
                  unsigned int *ecx, unsigned int *edx)
{
    asm volatile("cpuid"
        : "=a" (*eax), "=b" (*ebx), "=c" (*ecx), "=d" (*edx) : "a" (fn), "c" (*ecx));
}

int is_avx_supported()
{
    unsigned int eax, ebx, ecx, edx;

    // In: EAX=1, Out: ECX[bit 28]
    cpuid(1, &eax, &ebx, &ecx, &edx);
    return (ecx & (1 << 28)) ? 1 : 0;
}

int is_avx2_supported()
{
    unsigned int eax, ebx, ecx, edx;

    // In: EAX=7, ECX=0, Out: EBX[bit 5]
    ecx = 0;
    cpuid(7, &eax, &ebx, &ecx, &edx);
    return (ebx & (1 << 5)) ? 1 : 0;
}

int main()
{
    printf("AVX supported: %d\n", is_avx_supported());
    printf("AVX2 supported: %d\n", is_avx2_supported());
    
    return 0;
}    