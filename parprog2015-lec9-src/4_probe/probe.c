#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define NELEMS(x) (sizeof(x) / sizeof((x)[0]))

int main(int argc, char **argv)
{
    int rank, commsize, count;    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);

    if (rank == 0) {
        float buf[10];
        for (int i = 0; i < 10; i++)
            buf[i] = i * 10.0;
        MPI_Send(buf, 10, MPI_FLOAT, 2, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        int buf[6];
        for (int i = 0; i < 6; i++)
            buf[i] = i * 2 + 1;
        MPI_Send(buf, 6, MPI_INT, 2, 1, MPI_COMM_WORLD);
    } else if (rank == 2) {
        MPI_Status status;
        for (int m = 0; m < 2; m++) {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // Wait for incomming message
            if (status.MPI_TAG == 0) {
                MPI_Get_count(&status, MPI_FLOAT, &count);
                float *buf = malloc(sizeof(*buf) * count);
                MPI_Recv(buf, count, MPI_FLOAT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);
                printf("Master received: ");
                for (int i = 0; i < count; i++)
                    printf("%.2f ", buf[i]);
                printf("\n");
                free(buf);
            } else if (status.MPI_TAG == 1) {
                MPI_Get_count(&status, MPI_INT, &count);
                int *buf = malloc(sizeof(*buf) * count);
                MPI_Recv(buf, count, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);
                printf("Master received: ");
                for (int i = 0; i < count; i++)
                    printf("%d ", buf[i]);
                printf("\n");
                free(buf);
            }
        }            
    }

    MPI_Finalize();
    return 0;
}
