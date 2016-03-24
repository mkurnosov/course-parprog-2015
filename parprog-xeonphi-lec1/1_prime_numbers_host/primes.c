#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <offload.h>

const int a = 1;
const int b = 3000000;

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

double run_host_serial()
{
    double t = wtime();
    int n = count_prime_numbers(a, b);
    t = wtime() - t;

    printf("Result (host serial): %d\n", n);
    return t;
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
        /* Loop over odd numbers: a, a + 2, a + 4, ... , b */
        #pragma omp for schedule(dynamic, 100) reduction(+:nprimes)
        for (int i = a; i <= b; i += 2) {
            if (is_prime_number(i))
                nprimes++;
        }
    }
    return nprimes;
}

double run_host_parallel()
{
    double t = wtime();
    int n = count_prime_numbers_omp(a, b);
    t = wtime() - t;

    printf("Result (host parallel): %d\n", n);
    return t;
}

int main(int argc, char **argv)
{
    printf("Count prime numbers in [%d, %d]\n", a, b);
    double thost_serial = run_host_serial();
    double thost_par = run_host_parallel();
    
    printf("Execution time (host serial): %.6f\n", thost_serial);
    printf("Execution time (host parallel): %.6f\n", thost_par);
    printf("Speedup host_serial/host_omp: %.2f\n", thost_serial / thost_par);
    
    return 0;
}
