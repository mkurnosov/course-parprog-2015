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

void get_chunk(int a, int b, int commsize, int rank, int *lb, int *ub)
{
    /* 
     * This algorithm is based on OpenMP 4.0 spec (Sec. 2.7.1, default schedule for loops)
     * For a team of commsize processes and a sequence of n items, let ceil(n ⁄ commsize) be the integer q 
     * that satisfies n = commsize * q - r, with 0 <= r < commsize.
     * Assign q iterations to the first commsize - r processes, and q - 1 iterations to the remaining r processes.
     */
    int n = b - a + 1;
    int q = n / commsize;
    if (n % commsize)
        q++;
    int r = commsize * q - n;
    
    /* Compute chunk size for the process */
    int chunk = q;
    if (rank >= commsize - r)
        chunk = q - 1;
    
    /* Determine start item for the process */
    *lb = a;
    if (rank > 0) {
        /* Count sum of previous chunks */
        if (rank <= commsize - r)
            *lb += q * rank;
        else
            *lb += q * (commsize - r) + (q - 1) * (rank - (commsize - r));
    }
    *ub = *lb + chunk - 1;
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

int count_prime_numbers_par(int a, int b)
{
    int nprimes = 0;    

    int lb, ub;
    get_chunk(a, b, get_comm_size(), get_comm_rank(), &lb, &ub);
            
    /* Count '2' as a prime number */
    if (lb <= 2) {
        nprimes = 1;
        lb = 2;
    }        
        
    /* Shift 'a' to odd number */
    if (lb % 2 == 0)
        lb++;
        
    /* Loop over odd numbers: a, a + 2, a + 4, ... , b */
    for (int i = lb; i <= ub; i += 2) {
        if (is_prime_number(i))
            nprimes++;
    }
    
    int nprimes_global;
    MPI_Reduce(&nprimes, &nprimes_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);    
    return nprimes_global;
}

double run_serial()
{
    double t = MPI_Wtime();
    int n = count_prime_numbers(a, b);
    t = MPI_Wtime() - t;
    
    printf("Result (serial): %d\n", n);
    return t;
}

double run_parallel()
{    
    double t = MPI_Wtime();
    int n = count_prime_numbers_par(a, b);
    t = MPI_Wtime() - t;
    printf("Process %d/%d execution time: %.6f\n", get_comm_rank(), get_comm_size(), t);

    if (get_comm_rank() == 0)
        printf("Result (parallel): %d\n", n);
    return t;
}

int main(int argc, char **argv)
{    
    MPI_Init(&argc, &argv);

    // Start serial version
    double tserial = 0;
    if (get_comm_rank() == 0)
        tserial = run_serial();

    // Start parallel version
    double tparallel = run_parallel();
        
    if (get_comm_rank() == 0) {
        printf("Count prime numbers on [%d, %d]\n", a, b);
        printf("Execution time (serial): %.6f\n", tserial);
        printf("Execution time (parallel): %.6f\n", tparallel);
        printf("Speedup (processes %d): %.2f\n", get_comm_size(), tserial / tparallel);
    }
    
    MPI_Finalize();
    return 0;
}

