#include <mpi.h>
#include <stdio.h>

#define NELEMS(x) (sizeof(x) / sizeof((x)[0]))

int main(int argc, char **argv)
{
    int rank, commsize;
    float buf[100];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);

    if (rank == 0) {
        for (int i = 0; i < NELEMS(buf); i++)
            buf[i] = (float)i;
        MPI_Send(buf, NELEMS(buf), MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Status status;
        MPI_Recv(buf, 10, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Master received: ");
        int count;
        MPI_Get_count(&status, MPI_FLOAT, &count);
        for (int i = 0; i < count; i++)
            printf("%f ", buf[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
