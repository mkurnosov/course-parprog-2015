#include <stdio.h>
#include <math.h>
#include <omp.h>

enum {
    n = 1000000
};

float x[n], y[n];

float f(float x)
{
    return pow(sin(x), 2.0) * 0.5;
}

void do_stuff()
{   
    printf("Thread %d: do_stuff\n", omp_get_thread_num());
}

void do_stuff_last()
{   
    printf("Thread %d: do_stuff_last\n", omp_get_thread_num());
}

void fun()
{    
    #pragma omp parallel
    {
        // Parallel code
        #pragma omp for nowait
        for (int i = 0; i < n; i++)
            x[i] = f(i);

        // Serial code 
        #pragma omp single
        do_stuff();
        
        #pragma omp barrier
        /* Ждем готовности x[0:n-1] */

        // Parallel code
        #pragma omp for nowait
        for (int i = 0; i < n; i++)
            y[i] = x[i] + 2.0 * f(i);
        
        // Serial code
        #pragma omp master
        do_stuff_last();
    }        
}

int main(int argc, char **argv)
{
    fun();
    return 0;
}
