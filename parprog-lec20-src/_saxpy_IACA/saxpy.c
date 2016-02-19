#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include <sys/time.h>
#include <iacaMarks.h>

enum {
    n = 1000000 
};

void saxpy(float *x, float *y, float a, int n)
{
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}

void saxpy_sse(float * restrict x, float * restrict y, float a, int n)
{
    __m128 *xx = (__m128 *)x;
    __m128 *yy = (__m128 *)y;
   
    int k = n / 4;
    __m128 aa = _mm_set_ps1(a);
    for (int i = 0; i < k; i++) {
        IACA_START
        __m128 z = _mm_mul_ps(aa, xx[i]);          
        yy[i] = _mm_add_ps(z, yy[i]);
        IACA_END
    }
}

void *xmalloc(size_t size)
{
    void *p = malloc(size);
    if (!p) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }
    return p;
}

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

double run_serial()
{
    float *x, *y, a = 2.0;

    /* Run serial version */    
    x = xmalloc(sizeof(*x) * n);
    y = xmalloc(sizeof(*y) * n);
    for (int i = 0; i < n; i++) {
        x[i] = i * 2 + 1.0;
        y[i] = i;
    }
    
    double t = wtime();
    saxpy(x, y, a, n);
    t = wtime() - t;    
    
    printf("val = %f\n", y[0]);

    printf("Elapsed time (scalar): %.6f sec.\n", t);
    free(x);
    free(y);    
    return t;
}

double run_vectorized()
{
    float *x, *y, a = 2.0;

    /* Run parallel version */    
    x = _mm_malloc(sizeof(*x) * n, 16);
    y = _mm_malloc(sizeof(*y) * n, 16);
    for (int i = 0; i < n; i++) {
        x[i] = i * 2 + 1.0;
        y[i] = i;
    }
    
    double t = wtime();
    saxpy_sse(x, y, a, n);
    t = wtime() - t;
    
    printf("val = %f\n", y[0]);
        
    printf("Elapsed time (SSE): %.6f sec.\n", t);
    free(x);
    free(y);    
    return t;
}

int main(int argc, char **argv)
{
    printf("SAXPY (y[i] = a * x[i] + y[i]; n = %d)\n", n);
    double tserial = run_serial();
    double tsse = run_vectorized();
    
    printf("Speedup: %.2f\n", tserial / tsse);
        
    return 0;
}
