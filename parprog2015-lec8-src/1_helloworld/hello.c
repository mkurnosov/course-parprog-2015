#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) 
{
    int commsize, rank, len;
    char procname[MPI_MAX_PROCESSOR_NAME];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(procname, &len);
    
    printf("Hello, MPI World! Process %d of %d on node %s.\n", rank, commsize, procname);
        
    MPI_Finalize();
    return 0;
}
