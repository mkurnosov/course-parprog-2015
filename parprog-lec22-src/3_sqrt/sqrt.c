#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <xmmintrin.h>
#include <sys/time.h>

#define EPS 1E-6

enum {
    n = 1000003
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

void compute_sqrt_sse(float *in, float *out, int n)
{
    __m128 *in_vec = (__m128 *)in;
    __m128 *out_vec = (__m128 *)out;
    int k = n / 4;
                    
    __m128 zero = _mm_setzero_ps();
    for (int i = 0; i < k; i++) {
        // 1. Compute sqrt: all zero elements will be filled with NaNs
        //
        // 2. Vector compare (greater-than): in[i] > 0
        //    v = [v3, v2, v1, v0]
        //    mask = cmpgt(v, zero) ==> mask = [v3 > 0 ? 0xffffffff : 0 | v2 > 0 ? 0xffffffff : 0 | ... | v0 > 0 ? 0xffffffff : 0]
        //    mask = cmpgt([7, 1, 0, 2], zero) ==> mask = [0xffffffff, 0xffffffff, 0, 0xffffffff]
        // 
        // 3. Masked assignment -- copy elements for case: v[i] > 0
        //    gtzero_vec = and(mask, sqrt_vec) ==> all NaNs will be replaced by 0
        //    gtzero_vec = and([0xffffffff, 0xffffffff, 0, 0xffffffff], [s3, s2, NaN, s0]) = [s3, s2, 0, s0]
        // 
        // 4. Masked assignment -- copy elements for case: v[i] <= 0
        //    lezero_vec = andnot(mask, zero) ==> lezero_vec[i] = NOT mask[i] AND zero[i]
        //    lezero_vec = andnot([0xffffffff, 0xffffffff, 0, 0xffffffff], [0f, 0f, 0f, 0f]) = [0, 0, 0f, 0]
        //
        // 5. Combine results for both cases
        //    out_vec = or(gezero_vec, lezero_vec)
        //    out_vec = or([s3, s2, 0, s0], [0, 0, 0f, 0]) = [s3, s2, 0f, s0]
        //    
        __m128 v = _mm_load_ps((float *)&in_vec[i]);
        __m128 sqrt_vec = _mm_sqrt_ps(v);
        __m128 mask = _mm_cmpgt_ps(v, zero); 
        __m128 gtzero_vec = _mm_and_ps(mask, sqrt_vec);  
        __m128 lezero_vec = _mm_andnot_ps(mask, zero);
        out_vec[i] = _mm_or_ps(gtzero_vec, lezero_vec);
    }

    for (int i = k * 4; i < n; i++)
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
    float *in = _mm_malloc(sizeof(*in) * n, 16);
    float *out = _mm_malloc(sizeof(*out) * n, 16);
    srand(0);
    for (int i = 0; i < n; i++) {
        in[i] = rand() > RAND_MAX / 2 ? 0 : rand() / (float)RAND_MAX * 1000.0;
    }
    double t = wtime();
    compute_sqrt_sse(in, out, n);
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
