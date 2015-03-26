#include <stdio.h>
#include <omp.h>

enum {
    M = 2,
    N = 5
};

float m[M * N];

void fun1()
{
    // 1D by horizontal strips
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                m[i * N + j] = i * j;
            }
        }
    }        
}

void fun2()
{
    // 1D by vertical strips
    #pragma omp parallel
    {
        printf("Threads %d\n", omp_get_num_threads());

        for (int i = 0; i < M; i++) {
            #pragma omp for
            for (int j = 0; j < N; j++) {
                m[i * N + j] = i * j;
            }
        }
    }        
}

void fun3()
{
    #pragma omp parallel
    {
        printf("Threads %d\n", omp_get_num_threads());

        #pragma omp for collapse(2)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d [%d, %d]\n", omp_get_thread_num(), i, j);
                m[i * N + j] = i * j;
            }
        }
    }        
}

void fun4()
{
    #pragma omp parallel
    {
        printf("Threads %d\n", omp_get_num_threads());

        // 3 3 2 2
        #pragma omp for
        for (int ij = 0; ij < M * N; ij++) {
            int i = ij / N;
            int j = ij % N;
            printf("%d [%d, %d]\n", omp_get_thread_num(), i, j);
            m[i * N + j] = i * j;
        }
    }        
}

int main(int argc, char **argv)
{
    fun3();
    printf("---\n");
    fun4();
    return 0;
}
