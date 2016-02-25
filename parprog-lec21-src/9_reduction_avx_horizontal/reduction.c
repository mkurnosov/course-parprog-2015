#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <sys/time.h>

enum {
    n = 1000003
};

double sum(double *v, int n)
{
    double s = 0;
    for (int i = 0; i < n; i++)
        s += v[i];
    return s;
}

double sum_avx(double * restrict v, int n)
{
    __m256d *vv = (__m256d *)v;    
    int k = n / 4;
    __m256d sumv = _mm256_setzero_pd();
    for (int i = 0; i < k; i++) {
        sumv = _mm256_add_pd(sumv, vv[i]);
    }
    
    // Compute s = sumv[0] + sumv[1] + sumv[2] + sumv[3]
    // AVX _mm256_hadd_pd:
    //   _mm256_hadd_pd(a, a) => a = [a3 + a2 | a3 + a2 | a1 + a0 | a1 + a0]
    sumv = _mm256_hadd_pd(sumv, sumv);
    // Permute high and low 128 bits of sumv: [a1 + a0 | a1 + a0 | a3 + a2 | a3 + a2]
    __m256d sumv_permuted = _mm256_permute2f128_pd(sumv, sumv, 1);
    // sumv = [a1 + a0 + a3 + a2 | --//-- | ...]
    sumv = _mm256_add_pd(sumv_permuted, sumv);    

    double t[4] __attribute__ ((aligned (16)));   
    _mm256_store_pd(t, sumv); 
    double s = t[0];
    //double s = t[0] + t[1] + t[2] + t[3];
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
    double *v = xmalloc(sizeof(*v) * n);
    for (int i = 0; i < n; i++)
        v[i] = i + 1.0;
    
    double t = wtime();
    double res = sum(v, n);
    t = wtime() - t;    
    
    double valid_result = (1.0 + (double)n) * 0.5 * n;
    printf("Result (scalar): %.6f err = %f\n", res, fabsf(valid_result - res));
    printf("Elapsed time (scalar): %.6f sec.\n", t);
    free(v);
    return t;
}

double run_vectorized()
{
    double *v = _mm_malloc(sizeof(*v) * n, 32);
    for (int i = 0; i < n; i++)
        v[i] = i + 1.0;
    
    double t = wtime();
    double res = sum_avx(v, n);
    t = wtime() - t;    
    
    double valid_result = (1.0 + (double)n) * 0.5 * n;
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
