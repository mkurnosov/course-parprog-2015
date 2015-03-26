#include <stdio.h>
#include <omp.h>

int main()
{
    int n = 20;
        
    #pragma omp parallel for ordered
    for (int i = 0; i < n; i++) {
        #pragma omp ordered
        printf("%d ", omp_get_thread_num());
    }
    printf("\n");

    #pragma omp parallel for schedule(static, 1) ordered
    for (int i = 0; i < n; i++) {
        #pragma omp ordered
        printf("%d ", omp_get_thread_num());
    }
    printf("\n");

    #pragma omp parallel for schedule(static, 3) ordered
    for (int i = 0; i < n; i++) {
        #pragma omp ordered
        printf("%d ", omp_get_thread_num());
    }
    printf("\n");

    #pragma omp parallel for schedule(dynamic, 1) ordered
    for (int i = 0; i < n; i++) {
        #pragma omp ordered
        printf("%d ", omp_get_thread_num());
    }
    printf("\n");
    
    #pragma omp parallel for schedule(guided, 1) ordered
    for (int i = 0; i < n; i++) {
        #pragma omp ordered
        printf("%d ", omp_get_thread_num());
    }
    printf("\n");

    return 0;
}
