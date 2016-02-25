#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pmmintrin.h>
#include <sys/time.h>

enum {
    n = 1000003
};

float sum(float *v, int n)
{
    float s = 0;
    for (int i = 0; i < n; i++)
        s += v[i];
    return s;
}

float sum_sse(float * restrict v, int n)
{
    __m128 *vv = (__m128 *)v;    
    int k = n / 4;
    __m128 sumv = _mm_setzero_ps();
    for (int i = 0; i < k; i++) {
        sumv = _mm_add_ps(sumv, vv[i]);
    }
    
    // Compute s = sumv[0] + sumv[1] + sumv[2] + sumv[3]
    // SSE3 horizontal operation:
    //   hadd(a, a) => a = [a3 + a2 | a1 + a0 | a3 + a2 | a1 + a0] 
    //   hadd: [a3 + a2 + a1 + a0 | --/-- | --/-- | --/-- ]
    sumv = _mm_hadd_ps(sumv, sumv);
    sumv = _mm_hadd_ps(sumv, sumv);
    float s __attribute__ ((aligned (16))) = 0;   
    _mm_store_ss(&s, sumv);   
    
    for (int i = k * 4; i < n; i++)
        s += v[i];
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
    float *v = xmalloc(sizeof(*v) * n);
    for (int i = 0; i < n; i++)
        v[i] = i + 1.0;
    
    double t = wtime();
    float res = sum(v, n);
    t = wtime() - t;    
    
    float valid_result = (1.0 + (float)n) * 0.5 * n;
    printf("Result (scalar): %.6f err = %f\n", res, fabsf(valid_result - res));
    printf("Elapsed time (scalar): %.6f sec.\n", t);
    free(v);
    return t;
}

double run_vectorized()
{
    float *v = _mm_malloc(sizeof(*v) * n, 16);
    for (int i = 0; i < n; i++)
        v[i] = i + 1.0;
    
    double t = wtime();
    float res = sum_sse(v, n);
    t = wtime() - t;    
    
    float valid_result = (1.0 + (float)n) * 0.5 * n;
    printf("Result (vectorized): %.6f err = %f\n", res, fabsf(valid_result - res));
    printf("Elapsed time (vectorized): %.6f sec.\n", t);
    free(v);
    return t;
}

int main(int argc, char **argv)
{
    printf("Reduction: n = %d\n", n);
    double tscalar = run_scalar();
    double tvec = run_vectorized();
    
    printf("Speedup: %.2f\n", tscalar / tvec);
        
    return 0;
}
