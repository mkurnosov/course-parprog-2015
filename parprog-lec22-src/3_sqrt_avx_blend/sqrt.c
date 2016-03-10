#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <immintrin.h>
#include <sys/time.h>

#define EPS 1E-6

enum {
    n = 1000007
};

void compute_sqrt(float *in, float *out, int n)
{
    for (int i = 0; i < n; i++) {
        if (in[i] > 0)
            out[i] = sqrtf(in[i]);
        else
            out[i] = 0.0;
    }
}

void compute_sqrt_avx(float *in, float *out, int n)
{
    __m256 *in_vec = (__m256 *)in;
    __m256 *out_vec = (__m256 *)out;
    int k = n / 8;
                    
    __m256 zero = _mm256_setzero_ps();
    for (int i = 0; i < k; i++) {
        // 1. Compute sqrt: all zero elements will be filled with NaNs
        //
        // 2. Vector compare (greater-than): in[i] > 0
        //    mask = cmpgt([7, 1, 0, 2, 0, 4, 4, 9], zero) ==> mask = [0xffffffff, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, ..., 0xffffffff]
        // 
        // 3. Blend (merge results from two vectors by mask)
        //    blend(zero, [7, 1, 0, 2, 0, 4, 4, 9], mask) = [7, 1, 0f, 2, 0f, 4, 4, 9]
        //
        __m256 v = _mm256_load_ps((float *)&in_vec[i]);
        __m256 sqrt_vec = _mm256_sqrt_ps(v);
        __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);        
        out_vec[i] = _mm256_blendv_ps(zero, sqrt_vec, mask);
    }

    for (int i = k * 8; i < n; i++)
        out[i] = in[i] > 0 ? sqrtf(in[i]) : 0.0;        
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
    float *in = xmalloc(sizeof(*in) * n);
    float *out = xmalloc(sizeof(*out) * n);
    srand(0);
    for (int i = 0; i < n; i++) {
        in[i] = rand() > RAND_MAX / 2 ? 0 : rand() / (float)RAND_MAX * 1000.0;
    }
    double t = wtime();
    compute_sqrt(in, out, n);
    t = wtime() - t;    
    
#if 0
    for (int i = 0; i < n; i++)
        printf("%.4f ", out[i]);
    printf("\n");        
#endif
    
    printf("Elapsed time (scalar): %.6f sec.\n", t);
    free(in);
    free(out);
    return t;
}

double run_vectorized()
{
    float *in = _mm_malloc(sizeof(*in) * n, 32);
    float *out = _mm_malloc(sizeof(*out) * n, 32);
    srand(0);
    for (int i = 0; i < n; i++) {
        in[i] = rand() > RAND_MAX / 2 ? 0 : rand() / (float)RAND_MAX * 1000.0;
    }
    double t = wtime();
    compute_sqrt_avx(in, out, n);
    t = wtime() - t;    

#if 0
    for (int i = 0; i < n; i++)
        printf("%.4f ", out[i]);
    printf("\n");        
#endif
    
    for (int i = 0; i < n; i++) {
        float r = in[i] > 0 ? sqrtf(in[i]) : 0.0;
        if (fabs(out[i] - r) > EPS) {
            fprintf(stderr, "Verification: FAILED at out[%d] = %f != %f\n", i, out[i], r);
            break;
        }
    }
    
    printf("Elapsed time (vectorized): %.6f sec.\n", t);
    free(in);
    free(out);
    return t;
}

int main(int argc, char **argv)
{
    printf("Tabulate sqrt: n = %d\n", n);
    double tscalar = run_scalar();
    double tvec = run_vectorized();
    
    printf("Speedup: %.2f\n", tscalar / tvec);
        
    return 0;
}
