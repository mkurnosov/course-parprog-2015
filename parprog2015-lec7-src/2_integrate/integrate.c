#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

const double eps = 1E-24;
const double threshold = 0.05;
const double pi_real = 3.141592653589793238462643;

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

double f(double x) 
{
    return 4.0 / (1.0 + x * x);
}

double integrate(double left, double right, double f_left, double f_right, double leftright_area)
{
    double mid = (left + right) / 2;
    double f_mid = f(mid);
    double left_area = (f_left + f_mid) * (mid - left) / 2;
    double right_area = (f_mid + f_right) * (right - mid) / 2;
    if (fabs((left_area + right_area) - leftright_area) > eps) {
        left_area = integrate(left, mid, f_left, f_mid, left_area);
        right_area = integrate(mid, right, f_mid, f_right, right_area);
    }
    return left_area + right_area;
}

double integrate_omp(double left, double right, double f_left, double f_right, double leftright_area)
{
    double mid = (left + right) / 2;
    double f_mid = f(mid);
    double left_area = (f_left + f_mid) * (mid - left) / 2;
    double right_area = (f_mid + f_right) * (right - mid) / 2;
    if (fabs((left_area + right_area) - leftright_area) > eps) {
        if (right - left < threshold) {
            return integrate(left, right, f_left, f_right, leftright_area);
        }
        #pragma omp task shared(left_area)
        {
            left_area = integrate_omp(left, mid, f_left, f_mid, left_area);
        }
        right_area = integrate_omp(mid, right, f_mid, f_right, right_area);
        #pragma omp taskwait
    }
    return left_area + right_area;
}

double run_serial()
{       
    double t = wtime();
    double pi = integrate(0.0, 1.0, f(0), f(1), (f(0) + f(1)) / 2);
    t = wtime() - t;
    printf("Result (serial): %.4f; error %.12f\n", pi, fabs(pi - pi_real));
    return t;
}

double run_parallel()
{
    double pi;
    double t = wtime();
    #pragma omp parallel
    {
        #pragma omp single nowait
        pi = integrate_omp(0.0, 1.0, f(0), f(1), (f(0) + f(1)) / 2);
    }        
    t = wtime() - t;
    printf("Result (parallel): %.4f; error %.12f\n", pi, fabs(pi - pi_real));
    return t;
}

int main(int argc, char **argv)
{
    printf("Integration by trapezoidal rule\n");
    double tserial = run_serial();
    double tparallel = run_parallel();
    
    printf("Execution time (serial): %.6f\n", tserial);
    printf("Execution time (parallel): %.6f\n", tparallel);
    printf("Speedup: %.2f\n", tserial / tparallel);
    
    return 0;
}

