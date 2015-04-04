#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

enum {
    N = 1000000
};

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

double sum(double *v, int low, int high)
{
    if (low == high)
        return v[low];
    int mid = (low + high) / 2;
    return sum(v, low, mid) + sum(v, mid + 1, high);
}

double sum_omp(double *v, int low, int high)
{    
    static int nthreads_used = 1;
    
    if (low == high)
        return v[low];
    
    double sum_left, sum_right;
    int mid = (low + high) / 2;

    if (nthreads_used >= omp_get_thread_limit()) {
        return sum(v, low, high);
    }        
    
    #pragma omp atomic
    nthreads_used++;

    #pragma omp parallel num_threads(2)
    {
        #pragma omp sections
        {
            #pragma omp section
            sum_left = sum_omp(v, low, mid);

            #pragma omp section
            sum_right = sum_omp(v, mid + 1, high);
        }
    }
    return sum_left + sum_right;
}

int ilog2(int x)
{
    return log(x) / log(2.0);
}

double run_serial()
{
    double *v = malloc(sizeof(*v) * N);
    for (int i = 0; i < N; i++)
        v[i] = i + 1.0;
        
    double t = wtime();
    double res = sum(v, 0, N - 1);
    t = wtime() - t;
    printf("Result (serial): %.4f; error %.12f\n", res, fabs(res - (1.0 + N) / 2.0 * N));
    free(v);
    return t;
}

double run_parallel()
{
    double *v = malloc(sizeof(*v) * N);
    for (int i = 0; i < N; i++)
        v[i] = i + 1.0;
  
    omp_set_nested(1);
    printf("Parallel version:\n");
    printf("  OMP_THREAD_LIMIT = %d\n", omp_get_thread_limit());
    printf("  OMP_NESTED = %d\n", omp_get_nested());
            
    double t = wtime();
    double res = sum_omp(v, 0, N - 1);
    t = wtime() - t;
    printf("Result (parallel): %.4f; error %.12f\n", res, fabs(res - (1.0 + N) / 2.0 * N));
    free(v);
    return t;
}

int main(int argc, char **argv)
{
    printf("Recursive summation N = %d\n", N);
    double tserial = run_serial();
    double tparallel = run_parallel();
    
    printf("Execution time (serial): %.6f\n", tserial);
    printf("Execution time (parallel): %.6f\n", tparallel);
    printf("Speedup: %.2f\n", tserial / tparallel);
    
    return 0;
}
