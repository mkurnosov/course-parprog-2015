#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <xmmintrin.h>
#include <sys/time.h>

enum {
    n = 1000003
};

int main(int argc, char **argv)
{
    float *a = malloc(sizeof(*a) * n);
    float *b = malloc(sizeof(*b) * n);
    float *c = malloc(sizeof(*c) * n);
    
    #pragma GCC no_vector
    for (int i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }
    
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
        
    printf("Elem %f\n", c[0]);
    free(c);
    free(b);
    free(a);        
    return 0;
}
