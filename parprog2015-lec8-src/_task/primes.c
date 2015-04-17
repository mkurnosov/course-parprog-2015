#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

const int a = 1;
const int b = 10000000;

int get_comm_rank()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int get_comm_size()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
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

int count_prime_numbers_par(int a, int b)
{
    /* TODO */
}

double run_parallel()
{    
    double t = MPI_Wtime();
    int n = count_prime_numbers_par(a, b);
    t = MPI_Wtime() - t;

    if (get_comm_rank() == 0)
        printf("Result (parallel): %d\n", n);

    double tmax;
    MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return tmax;
}

int main(int argc, char **argv)
{    
    MPI_Init(&argc, &argv);

    // Start parallel version
    double tparallel = run_parallel();
        
    if (get_comm_rank() == 0) {
        printf("Count prime numbers on [%d, %d]\n", a, b);
        printf("Execution time (%d processes): %.6f\n", get_comm_size(), tparallel);
    }
    
    MPI_Finalize();
    return 0;
}
