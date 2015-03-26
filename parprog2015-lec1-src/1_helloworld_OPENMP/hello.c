#include <stdio.h>
#include <omp.h>

int main(int argc, char **argv)
{
#ifdef _OPENMP
    #pragma omp parallel num_threads(4)
    {
        printf("Hello, multithreaded world: thread %d of %d\n", 
               omp_get_thread_num(), omp_get_num_threads());        
    }
    printf("OpenMP version %d\n", _OPENMP);
    if (_OPENMP >= 201107)
        printf("  OpenMP 3.1 is supported\n");        
#endif
    return 0;
}
