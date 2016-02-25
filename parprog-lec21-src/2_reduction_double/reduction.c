#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <xmmintrin.h>
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

void sum_sse(double * restrict v, int n)
{
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
    return 0.0;
}

int main(int argc, char **argv)
{
    printf("Reduction: n = %d\n", n);
    double tscalar = run_scalar();
    double tvec = run_vectorized();
    
    printf("Speedup: %.2f\n", tscalar / tvec);
        
    return 0;
}
