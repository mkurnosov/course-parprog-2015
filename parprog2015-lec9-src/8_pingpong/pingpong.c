#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

enum {
    BUFSIZE = 1,
    NREPS = 100
};

int main(int argc, char **argv)
{
    int rank;
    char sbuf[BUFSIZE], rbuf[BUFSIZE];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
        
    double t = 0;
    if (rank == 0) {
        for (int i = 0; i < NREPS; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            t -= MPI_Wtime();
            MPI_Send(sbuf, BUFSIZE, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(rbuf, BUFSIZE, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            t += MPI_Wtime();
        }            
        t /= NREPS; 
        printf("Send + recv avg.time (sec): %.6f\n", t);
    } else if (rank == 1) {
        for (int i = 0; i < NREPS; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            t -= MPI_Wtime();
            MPI_Recv(rbuf, BUFSIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(sbuf, BUFSIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            t += MPI_Wtime();
        }            
        t /= NREPS;
        printf("Recv + send avg.time (sec): %.6f\n", t);
    }

    MPI_Finalize();
    return 0;
}
