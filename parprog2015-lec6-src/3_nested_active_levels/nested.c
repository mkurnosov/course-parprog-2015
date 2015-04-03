#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void level3(int parent)
{
    #pragma omp parallel num_threads(2)
    {                
        #pragma omp critical        
        printf("L3: parent %d, thread %d / %d, level %d (nested regions %d)\n", 
               parent, omp_get_thread_num(), omp_get_num_threads(), omp_get_active_level(), omp_get_level());

        // OMP level 3 >= max_active_levels => 1 thread in region
        // We have now <= 2 * 3 threads
    }
}

void level2(int parent)
{
    #pragma omp parallel num_threads(3)
    {                
        #pragma omp critical
        printf("L2: parent %d, thread %d / %d, level %d (nested regions %d)\n", 
               parent, omp_get_thread_num(), omp_get_num_threads(), omp_get_active_level(), omp_get_level());

        // OMP level 2, we have <= 2 * 3 threads        
        level3(omp_get_thread_num());
    }       
}

void level1()
{
    #pragma omp parallel num_threads(2)
    {                
        #pragma omp critical
        printf("L1: thread %d / %d, level %d (nested regions %d)\n", 
               omp_get_thread_num(), omp_get_num_threads(), omp_get_active_level(), omp_get_level());

        // OMP level 1, we have 2 threads here
        level2(omp_get_thread_num());
    }       
}

int main(int argc, char **argv)
{
    printf("OMP_DYNAMIC: %d\n", omp_get_dynamic());
    printf("OMP_NESTED: %d\n", omp_get_nested());
    printf("OMP_THREAD_LIMIT: %d\n", omp_get_thread_limit());
    printf("OMP_MAX_ACTIVE_LEVELS: %d\n", omp_get_max_active_levels());
    printf("ActiveNestedParRegions: %d\n", omp_get_active_level());
    printf("NestedParRegions: %d\n", omp_get_level());

    omp_set_nested(1);
    omp_set_max_active_levels(2);
    level1();
    
    return 0;
}
