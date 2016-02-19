#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <xmmintrin.h>
#include <sys/time.h>

#define EPS 1E-6

enum {
    n = 1000000
};

void daxpy(double * restrict x, double * restrict y, double a, int n)
{
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}

void daxpy_sse(double * restrict x, double * restrict y, double a, int n)
{
    __m128d *xx = (__m128d *)x;
    __m128d *yy = (__m128d *)y;
   
    int k = n / 2;
    __m128d aa = _mm_set1_pd(a);
    for (int i = 0; i < k; i++) {
        __m128d z = _mm_mul_pd(aa, xx[i]);          
        yy[i] = _mm_add_pd(z, yy[i]);
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

double run_scalar()
{
    double *x, *y, a = 2.0;

    x = xmalloc(sizeof(*x) * n);
    y = xmalloc(sizeof(*y) * n);
    for (int i = 0; i < n; i++) {
        x[i] = i * 2 + 1.0;
        y[i] = i;
    }
    
    double t = wtime();
    daxpy(x, y, a, n);
    t = wtime() - t;    

    /* Verification */
    for (int i = 0; i < n; i++) {
        double xx = i * 2 + 1.0;
        double yy = a * xx + i;
        if (fabs(y[i] - yy) > EPS) {
            fprintf(stderr, "run_scalar: verification failed (y[%d] = %f != %f)\n", i, y[i], yy);
            break;
        }
    }
    
    printf("Elapsed time (scalar): %.6f sec.\n", t);
    free(x);
    free(y);    
    return t;
}

double run_vectorized()
{
    double *x, *y, a = 2.0;

    x = _mm_malloc(sizeof(*x) * n, 16);
    y = _mm_malloc(sizeof(*y) * n, 16);
    for (int i = 0; i < n; i++) {
        x[i] = i * 2 + 1.0;
        y[i] = i;
    }
    
    double t = wtime();
    daxpy_sse(x, y, a, n);
    t = wtime() - t;
    
    /* Verification */
    for (int i = 0; i < n; i++) {
        double xx = i * 2 + 1.0;
        double yy = a * xx + i;
        if (fabs(y[i] - yy) > EPS) {
            fprintf(stderr, "run_vectorized: verification failed (y[%d] = %f != %f)\n", i, y[i], yy);
            break;
        }
    }
        
    printf("Elapsed time (vectorized): %.6f sec.\n", t);
    free(x);
    free(y);    
    return t;
}

int main(int argc, char **argv)
{
    printf("daxpy (y[i] = a * x[i] + y[i]; n = %d)\n", n);
    double tscalar = run_scalar();
    double tvec = run_vectorized();
    
    printf("Speedup: %.2f\n", tscalar / tvec);
        
    return 0;
}
