#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <smmintrin.h>
#include <sys/time.h>

enum { n = 1000003 };

float sdot(float *x, float *y, int n)
{
    float s = 0;
    for (int i = 0; i < n; i++)
        s += x[i] * y[i];
    return s;
}

float sdot_sse(float *x, float *y, int n)
{
    __m128 *xx = (__m128 *)x;
    __m128 *yy = (__m128 *)y;
    int k = n / 4;
    float s = 0;
    for (int i = 0; i < k; i++) {
        __m128 v = _mm_mul_ps(xx[i], yy[i]);
        v = _mm_hadd_ps(v, v);
        v = _mm_hadd_ps(v, v);
        float temp __attribute__ ((aligned (16)));
        _mm_store_ss(&temp, v);
        s += temp;
    }
        
    for (int i = k * 4; i < n; i++)
        s += x[i] * y[i];
    return s;    
}

float sdot_sse_1hadd(float *x, float *y, int n)
{
    __m128 *xx = (__m128 *)x;
    __m128 *yy = (__m128 *)y;
    int k = n / 4;
    __m128 vsum = _mm_setzero_ps();
    for (int i = 0; i < k; i++) {
        __m128 v = _mm_mul_ps(xx[i], yy[i]); // v = [x3 * y3 | x2 * y2 | ... ]
        v = _mm_hadd_ps(v, v);  // [x3 * y3 + x2 * y2 | x1 * y1 + x0 * y0 | ...]
        vsum = _mm_add_ps(vsum, v);
    }
    vsum = _mm_hadd_ps(vsum, vsum);
    float s __attribute__ ((aligned (16)));
    _mm_store_ss(&s, vsum);

    for (int i = k * 4; i < n; i++)
        s += x[i] * y[i];
    return s;    
}

float sdot_sse_dp(float *x, float *y, int n)
{
    __m128 *xx = (__m128 *)x;
    __m128 *yy = (__m128 *)y;
    int k = n / 4;
    float s __attribute__ ((aligned (16))) = 0;
    for (int i = 0; i < k; i++) {
        __m128 v = _mm_dp_ps(xx[i], yy[i], 0xff);  // SSE4
        float t __attribute__ ((aligned (16)));
        _mm_store_ss(&t, v);
        s += t;
    }
        
    for (int i = k * 4; i < n; i++)
        s += x[i] * y[i];
    return s;    
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
    float *x = xmalloc(sizeof(*x) * n);
    float *y = xmalloc(sizeof(*y) * n);
    for (int i = 0; i < n; i++) {
        x[i] = 2.0;
        y[i] = 3.0;
    }        
    
    double t = wtime();
    float res = sdot(x, y, n);
    t = wtime() - t;    
    
    float valid_result = 2.0 * 3.0 * n;
    printf("Result (scalar): %.6f err = %f\n", res, fabsf(valid_result - res));
    printf("Elapsed time (scalar): %.6f sec.\n", t);
    free(x);
    free(y);
    return t;
}

double run_vectorized()
{
    float *x = _mm_malloc(sizeof(*x) * n, 16);
    float *y = _mm_malloc(sizeof(*y) * n, 16);
    for (int i = 0; i < n; i++) {
        x[i] = 2.0;
        y[i] = 3.0;
    }        
    
    double t = wtime();
    float res = sdot_sse(x, y, n);
    t = wtime() - t;    
    
    float valid_result = 2.0 * 3.0 * n;
    printf("Result (vectorized): %.6f err = %f\n", res, fabsf(valid_result - res));
    printf("Elapsed time (vectorized): %.6f sec.\n", t);
    free(x);
    free(y);
    return t;
}

int main(int argc, char **argv)
{
    printf("SDOT: n = %d\n", n);
    float tscalar = run_scalar();
    float tvec = run_vectorized();
    
    printf("Speedup: %.2f\n", tscalar / tvec);
        
    return 0;
}
