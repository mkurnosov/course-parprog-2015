#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>

void fun()
{
    int a = 100;
    int b = 200;
    int c = 300;
    int d = 400;
    static int sum = 0;    
    printf("Before parallel: a = %d, b = %d, c = %d, d = %d\n", a, b, c, d);
        
    #pragma omp parallel private(a) firstprivate(b) num_threads(2)
    {
        int tid = omp_get_thread_num();
        printf("Thread %d: a = %d, b = %d, c = %d, d = %d\n", tid, a, b, c, d);
        a = 1;
        b = 2;
        
        #pragma omp threadprivate(sum)
        sum++;
        
        #pragma omp for lastprivate(c)
        for (int i = 0; i < 100; i++)
            c = i;
        /* c=99 - has the value from last iteration */
        
        #pragma omp sections lastprivate(d)
        {
            #pragma omp section
            {
                d = 1;                    
            }
            #pragma omp section  
            {
                d = 2;
            }
        }
        /* d=3 -  has the value form lexically last section */
            
    }

    // a = 100, b = 200, c = 99, d = 2, sum = 1
    printf("After parallel: a = %d, b = %d, c = %d, d = %d\n", a, b, c, d);
}

int main(int argc, char **argv)
{
    fun();
    return 0;
}
