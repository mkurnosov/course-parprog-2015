#include <stdio.h>
#include <omp.h>

void fun()
{
    #pragma omp parallel
    {

        #pragma omp master
        {
            printf("Thread in master %d\n", omp_get_thread_num());
        }
        
        #pragma omp single
        {
            printf("Thread in single %d\n", omp_get_thread_num());        
        }
    }        
}

int main(int argc, char **argv)
{
    fun();
    return 0;
}
