#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

const double goldenratio = 1.618;   /* Static storage duration (.rodata) */
double vec[1000];                   /* Static storage duration (.bss) */
int counter = 100;                  /* Static storage duration (.data) */

double fun(int a)
{
    double b = 1.0;         /* Automatic storage duration (stack, register) */
    static double gsum = 0; /* Static storage duration (.data) */

    _Thread_local static double sumloc = 5; /* Thread storage duration (.tdata) */   
    _Thread_local static double bufloc;     /* Thread storage duration (.tbbs) */   

    double *v = malloc(sizeof(*v) * 100);   /* Allocated storage duration (Heap) */
    
    #pragma omp parallel num_threads(2)
    {
        int c = 1.0;                        /* Automatic storage duration */
        printf("%p %p\n", &b, &c);
        /* ... */
    }
            
    free(v);
    return b + gsum + sumloc + v[0] + a + sumloc + bufloc;
}

int main(int argc, char **argv)
{
    int a = 1;   
    printf("%f\n", fun(a));
    return 0;
}
