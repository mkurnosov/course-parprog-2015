#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

enum {
    N = 1000000000
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
    if (low == high)
        return v[low];
    
    double sum_left, sum_right;
    int mid = (low + high) / 2;

    #pragma omp parallel num_threads(2)
    {
        #pragma omp sections
        {
            #pragma omp section 
            {
                sum_left = sum_omp(v, low, mid);

                //printf("L: Level %d / %d, thread %d / %d range [%d..%d] = %f\n", 
                //       omp_get_active_level(), omp_get_level(), omp_get_thread_num(), omp_get_num_threads(), low, mid, sum_left);
            }
            
            #pragma omp section
            {
                sum_right = sum_omp(v, mid + 1, high);

                //printf("R: Level %d / %d, thread %d / %d range [%d..%d] = %f\n", 
                //       omp_get_active_level(), omp_get_level(), omp_get_thread_num(), omp_get_num_threads(), mid + 1, high, sum_right);
            }                
        }
    }
    return sum_left + sum_right;
}

double sum_omp_fixed_depth_static(double *v, int low, int high)
{
    if (low == high)
        return v[low];
    
    double sum_left, sum_right;
    int mid = (low + high) / 2;

    if (omp_get_active_level() >= omp_get_max_active_levels())
        return sum_omp_fixed_depth_static(v, low, mid) + sum_omp_fixed_depth_static(v, mid + 1, high);    

    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();       
        if (tid == 0) {
            sum_left = sum_omp_fixed_depth_static(v, low, mid);
            //printf("L: Level %d / %d, thread %d / %d range [%d..%d] = %f\n", 
            //       omp_get_active_level(), omp_get_level(), omp_get_thread_num(), omp_get_num_threads(), low, mid, sum_left);
        } else if (tid == 1) {
            sum_right = sum_omp_fixed_depth_static(v, mid + 1, high);
            //printf("R: Level %d / %d, thread %d / %d range [%d..%d] = %f\n", 
            //       omp_get_active_level(), omp_get_level(), omp_get_thread_num(), omp_get_num_threads(), mid + 1, high, sum_right);
        }
    }
    return sum_left + sum_right;
}

double sum_omp_fixed_depth(double *v, int low, int high)
{
    if (low == high)
        return v[low];
    
    double sum_left, sum_right;
    int mid = (low + high) / 2;

    if (omp_get_active_level() >= omp_get_max_active_levels())
        return sum_omp_fixed_depth(v, low, mid) + sum_omp_fixed_depth(v, mid + 1, high);    

    #pragma omp parallel num_threads(2)
    {
        #pragma omp sections
        {
            #pragma omp section
            sum_left = sum_omp_fixed_depth(v, low, mid);

            #pragma omp section
            sum_right = sum_omp_fixed_depth(v, mid + 1, high);
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

    int maxthreads = atoi(getenv("OMP_NUM_THREADS"));
    int maxlevels = ilog2(maxthreads) + 1;
    omp_set_nested(1);
    omp_set_max_active_levels(maxlevels);
    printf("Parallel version: max_threads = %d, max_levels = %d\n", maxthreads, maxlevels);    
        
    double t = wtime();
    double res = sum_omp_fixed_depth_static(v, 0, N - 1);
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
