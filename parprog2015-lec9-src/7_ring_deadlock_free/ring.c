#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

enum {    
    BUFSIZE = 1000000
};

char rbuf[BUFSIZE], sbuf[BUFSIZE];

int main(int argc, char **argv)
{
    int rank, commsize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    
    int prev = (rank - 1 + commsize) % commsize;
    int next = (rank + 1) % commsize;
    
    MPI_Sendrecv(&sbuf, BUFSIZE, MPI_CHAR, next, 0, &rbuf, BUFSIZE, MPI_CHAR, prev, 0, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Process %d received from %d\n", rank, prev);

    MPI_Finalize();
    return 0;
}
