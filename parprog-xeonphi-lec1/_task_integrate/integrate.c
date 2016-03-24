#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

const double PI = 3.14159265358979323846;

const double a = -4.0;
const double b = 4.0;
const int nsteps = 100000000;

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

/* integrate: Integrates by rectangle method (midpoint rule) */
double integrate_omp(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5)); 
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

/* integrate: Integrates by rectangle method (midpoint rule) */
double integrate_phi_omp(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;
    
    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5)); 
    sum *= h;
    return sum;    
}

double run_phi()
{
    // TODO: offload this code to Xeon Phi
    double res;
    double t = wtime();
    res = integrate_phi_omp(func, a, b, nsteps);
    t = wtime() - t;

    printf("Result (phi): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

int main(int argc, char **argv)
{
    printf("Integration f(x) on [%.12f, %.12f], nsteps = %d\n", a, b, nsteps);
    double tserial = run_serial();
    double tparallel = run_parallel();
    double tphi = run_phi();
    
    printf("Execution time (serial): %.6f\n", tserial);
    printf("Execution time (parallel): %.6f\n", tparallel);
    printf("Execution time (phi): %.6f\n", tphi);
    printf("Speedup (host): %.2f\n", tserial / tparallel);
    printf("Speedup (host_omp/phi): %.2f\n", tparallel / tphi);
    printf("Speedup (host_serial/phi): %.2f\n", tserial / tphi);
    
    return 0;
}
