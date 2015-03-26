/*
 * primes.c: Example of prime numbers counting in OpenMP.
 *
 * (C) 2015 Mikhail Kurnosov <mkurnosov@gmail.com>
 */
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

const int a = 1;
const int b = 10000000;

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

/* 
 * is_prime_number: Returns 1 if n is a prime number and 0 otherwise. 
 *                  This function uses trial division primality test.
 */
int is_prime_number(int n)
{
    int limit = sqrt(n) + 1;
    for (int i = 2; i <= limit; i++) {
        if (n % i == 0)
            return 0;
    }
    return (n > 1) ? 1 : 0;
}

int count_prime_numbers(int a, int b)
{
    int nprimes = 0;
        
    /* Count '2' as a prime number */
    if (a <= 2) {
        nprimes = 1;
        a = 2;
    }        
        
    /* Shift 'a' to odd number */
    if (a % 2 == 0)
        a++;
        
    /* Loop over odd numbers: a, a + 2, a + 4, ... , b */
    for (int i = a; i <= b; i += 2) {
        if (is_prime_number(i))
            nprimes++;
    }
    return nprimes;
}

int count_prime_numbers_omp(int a, int b)
{
    int nprimes = 0;
        
    /* Count '2' as a prime number */
    if (a <= 2) {
        nprimes = 1;
        a = 2;
    }        
        
    /* Shift 'a' to odd number */
    if (a % 2 == 0)
        a++;
        
    #pragma omp parallel    
    {
        int nloc = 0;
        
        /* Loop over odd numbers: a, a + 2, a + 4, ... , b */
        #pragma omp for
        for (int i = a; i <= b; i += 2) {
            if (is_prime_number(i))
                nloc++;
        }
        
        #pragma omp atomic
        nprimes += nloc;
    }
    return nprimes;
}

double run_serial()
{
    double t = wtime();
    int n = count_prime_numbers(a, b);
    t = wtime() - t;

    printf("Result (serial): %d\n", n);
    return t;
}

double run_parallel()
{
    double t = wtime();
    int n = count_prime_numbers_omp(a, b);
    t = wtime() - t;

    printf("Result (parallel): %d\n", n);
    return t;
}

int main(int argc, char **argv)
{
    printf("Count prime numbers on [%d, %d]\n", a, b);
    double tserial = run_serial();
    double tparallel = run_parallel();
    
    printf("Execution time (serial): %.6f\n", tserial);
    printf("Execution time (parallel): %.6f\n", tparallel);
    printf("Speedup: %.2f\n", tserial / tparallel);
    
    return 0;
}

