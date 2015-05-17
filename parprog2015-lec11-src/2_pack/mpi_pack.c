#include <stdio.h>
#include <inttypes.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, packsize, position;
    int a;
    double b;
    uint8_t packbuf[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
	    a = 15;
		b = 3.14;
		packsize = 0;
        /* Pack data into the buffer */
		MPI_Pack(&a, 1, MPI_INT, packbuf, 100, &packsize, MPI_COMM_WORLD);
		MPI_Pack(&b, 1, MPI_DOUBLE, packbuf, 100, &packsize, MPI_COMM_WORLD);
    }
    
	MPI_Bcast(&packsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(packbuf, packsize, MPI_PACKED, 0, MPI_COMM_WORLD);
	if (rank != 0) {
		position = 0;
        /* Unpack data */
		MPI_Unpack(packbuf, packsize, &position, &a, 1, MPI_INT, MPI_COMM_WORLD);
		MPI_Unpack(packbuf, packsize, &position, &b, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	}
	printf("Process %d unpacked %d and %lf\n", rank, a, b);

    MPI_Finalize( );
    return 0;
}
