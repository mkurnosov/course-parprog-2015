#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <immintrin.h>
#include <sys/time.h>

enum {
    n = 1000003
};

float find_max(float *v, int n)
{
    float max = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (v[i] > max)
            max = v[i];
    }
    return max;
}

float find_max_avx(float * restrict v, int n)
{
    __m256 *vv = (__m256 *)v;
    int k = n / 8;
                
    __m256 maxval = _mm256_set1_ps(-FLT_MAX);
    for (int i = 0; i < k; i++)
        maxval = _mm256_max_ps(maxval, vv[i]);
    
    // Horizontal max
    // a = [a7, a6, ..., a1, a0]
    // shuffle(a, a, _MM_SHUFFLE(2, 1, 0, 3)) ==> shuffles each 128 bit line [a6, a5, a4, a7 | a2, a1, a0, a3]
    maxval = _mm256_max_ps(maxval, _mm256_shuffle_ps(maxval, maxval, _MM_SHUFFLE(2, 1, 0, 3))); 
    maxval = _mm256_max_ps(maxval, _mm256_shuffle_ps(maxval, maxval, _MM_SHUFFLE(2, 1, 0, 3))); 
    maxval = _mm256_max_ps(maxval, _mm256_shuffle_ps(maxval, maxval, _MM_SHUFFLE(2, 1, 0, 3))); 
    float t[8];
    _mm256_store_ps(t, maxval);
    float max = t[0] > t[5] ? t[0] : t[5]; 
    
    for (int i = k * 8; i < n; i++)
        if (v[i] > max)
            max = v[i];
    return max;
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
    float res = find_max(v, n);
    t = wtime() - t;    
    
    float valid_result = (float)n;
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
    float res = find_max_avx(v, n);
    t = wtime() - t;    
    
    float valid_result = find_max(v, n);
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
