/*
 * quicksort.c: Example of QucikSort in OpenMP.
 *
 * (C) 2015 Mikhail Kurnosov <mkurnosov@gmail.com>
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

enum {
    N = 2 * 1024 * 1024    
};

const int threshold = 1000;

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

void *xmalloc(size_t size)
{
    void *p = malloc(size);
    if (!p) {
        fprintf(stderr, "No enough memory\n");
        exit(EXIT_FAILURE);
    }
    return p;
}

int is_nondecreasing_sorted(double *v, int n)
{
    if (n < 2)
        return 1;

    // Non-deacreasing sorting: v[0] <= v[1] <= ... <= v[n - 1]
    for (int i = 1; i < n; i++) {
        if (v[i - 1] > v[i])
            return 0;
    }        
    return 1;
}

void swap(double *v, int i, int j)
{
    double temp = v[i];
    v[i] = v[j];
    v[j] = temp;
}

int partition(double *v, int low, int high)
{
    double pivot = v[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (v[j] <= pivot) {
            i++;
            swap(v, i, j);
        }
    }
    swap(v, i + 1, high);
    return i + 1;
}

/*
 * quicksort: Sorting n elements of array v in the non-decreasing order
 *            by quicksort algorithm (complexity in average case is O(n \log n)).
 */
void quicksort(double *v, int low, int high)
{
    if (low < high) {
        int k = partition(v, low, high);
        quicksort(v, low, k - 1);
        quicksort(v, k + 1, high);
    }
}

void quicksort_omp_tasks(double *v, int low, int high)
{
    if (low < high) {            
        if (high - low < threshold) {
            quicksort(v, low, high);
        } else {
            int k = partition(v, low, high);
            #pragma omp task
            quicksort_omp_tasks(v, low, k - 1);

            quicksort_omp_tasks(v, k + 1, high);
        }
    }
}

double run_serial()
{
    double *v = xmalloc(sizeof(*v) * N);
    
    srand(0);
    for (int i = 0; i < N; i++)
        v[i] = rand() % 1000;        
   
    double t = wtime();
    quicksort(v, 0, N - 1);
    t = wtime() - t;
    
    if (!is_nondecreasing_sorted(v, N)) {
        fprintf(stderr, "Verification FAILED (serial version)\n");
    }    
    free(v);
    return t;
} 

double run_parallel()
{
    double *v = xmalloc(sizeof(*v) * N);
    
    srand(0);
    for (int i = 0; i < N; i++)
        v[i] = rand() % 1000;        
   
    double t = wtime();
    #pragma omp parallel
    {
        #pragma omp single nowait
        quicksort_omp_tasks(v, 0, N - 1);
    }   
    t = wtime() - t;
    
    if (!is_nondecreasing_sorted(v, N)) {
        fprintf(stderr, "Verification FAILED (parallel version)\n");
    }    
    free(v);
    return t;
} 

int main() 
{
    printf("Soring by QuickSort, N = %d\n", N);
    double tserial = run_serial();
    double tparallel = run_parallel();
    
    printf("Execution time (serial): %.6f\n", tserial);
    printf("Execution time (parallel): %.6f\n", tparallel);
    printf("Speedup: %.2f\n", tserial / tparallel);

    return 0;
}

