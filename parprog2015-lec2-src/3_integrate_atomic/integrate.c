/*
 * integrate.c: Example of numerical integration in OpenMP.
 *
 * (C) 2015 Mikhail Kurnosov <mkurnosov@gmail.com>
 */
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

const double PI = 3.14159265358979323846;

const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

double func(double x)
{
    return exp(-x * x);
}

/* integrate: Integrates by rectangle method (midpoint rule) */
double integrate(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;
    
    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5)); 
    sum *= h;
    return sum;    
}

double run_serial()
{
    double t = wtime();
    double res = integrate(func, a, b, nsteps);
    t = wtime() - t;

    printf("Result (serial): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

double integrate_omp(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;    
    double sum = 0.0;
    
    #pragma omp parallel
    {        
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++) {
            double f = func(a + h * (i + 0.5));
            #pragma omp atomic
            sum += f;           // No atomics for FP-values, loop in asm: LD, ADD, Compare-And-Swap 
        }
    }
    sum *= h;
    return sum;
}

double run_parallel()
{
    double t = wtime();
    double res = integrate_omp(func, a, b, nsteps);
    t = wtime() - t;

    printf("Result (parallel): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

int main(int argc, char **argv)
{
    printf("Integration f(x) on [%.12f, %.12f], nsteps = %d\n", a, b, nsteps);
    double tserial = run_serial();
    double tparallel = run_parallel();
    
    printf("Execution time (serial): %.6f\n", tserial);
    printf("Execution time (parallel): %.6f\n", tparallel);
    printf("Speedup: %.2f\n", tserial / tparallel);
    
    return 0;
}
