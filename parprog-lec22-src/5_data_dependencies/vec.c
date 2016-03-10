#include <stdio.h>
#include <stdlib.h>

enum {
    n = 1000003
};

int main(int argc, char **argv)
{
    float *a = malloc(sizeof(*a) * n);
    float *b = malloc(sizeof(*b) * n);
    
    for (int i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }
    
    for (int i = 0; i < n - 1; i++) {
        a[i + 1] = a[i] + b[i];
    }
        
    printf("Elem %f\n", a[0]);
    free(a); 
    free(b);        
    return 0;
}
