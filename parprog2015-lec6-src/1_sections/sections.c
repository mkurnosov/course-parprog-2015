#define _POSIX_C_SOURCE  199309L

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void sleep_rand_ns(int min, int max)
{
    int delay = ((double)rand() / RAND_MAX) * (max - min + 1) + min;
    delay %= 999999999;
    nanosleep(&(struct timespec){.tv_sec = 0, .tv_nsec = delay}, NULL);
}

void sections()
{
    #pragma omp parallel num_threads(3)
    {        
        #pragma omp sections
        {
            // Section directive is optional for the first structured block
            {
                sleep_rand_ns(100000, 200000);
                printf("Section 0: thread %d / %d\n", omp_get_thread_num(), omp_get_num_threads());
            }
            
            #pragma omp section
            {
                sleep_rand_ns(100000, 200000);
                printf("Section 1: thread %d / %d\n", omp_get_thread_num(), omp_get_num_threads());                
            }
            
            #pragma omp section
            {
                sleep_rand_ns(100000, 200000);
                printf("Section 2: thread %d / %d\n", omp_get_thread_num(), omp_get_num_threads());
            }            

            #pragma omp section
            {
                sleep_rand_ns(100000, 200000);
                printf("Section 3: thread %d / %d\n", omp_get_thread_num(), omp_get_num_threads());
            }            
        }
    }       
}

void sections_static()
{
    printf("\n");
    #pragma omp parallel num_threads(3)
    {        
        int tid = omp_get_thread_num();
        switch (tid) {
            case 0:
                sleep_rand_ns(100000, 200000);
                printf("Section 0: thread %d / %d\n", omp_get_thread_num(), omp_get_num_threads());
                break;                                
            case 1:
                sleep_rand_ns(100000, 200000);
                printf("Section 1: thread %d / %d\n", omp_get_thread_num(), omp_get_num_threads());
                break;
            case 2:
                sleep_rand_ns(100000, 200000);
                printf("Section 3: thread %d / %d\n", omp_get_thread_num(), omp_get_num_threads());
                break;
            default:
                fprintf(stderr, "Error: TID > 2\n");
        }
    }        
}

int main(int argc, char **argv)
{
    sections();
    sections_static();            
    return 0;
}
