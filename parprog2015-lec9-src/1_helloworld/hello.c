#include <mpi.h>
#include <stdio.h>

#define NELEMS(x) (sizeof(x) / sizeof((x)[0]))

int main(int argc, char **argv)
{
    int rank, commsize, len, tag = 1;
    char host[MPI_MAX_PROCESSOR_NAME];
    char msg[128 + MPI_MAX_PROCESSOR_NAME];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Get_processor_name(host, &len);

    if (rank > 0) {
        /* Each process > 0 send message to the root */
        int count = snprintf(msg, NELEMS(msg), "Hello, I am %d of %d on %s", rank, commsize, host) + 1;
        MPI_Send(msg, count, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
    } else {
        MPI_Status status;
        printf("Hello, World. I am root (%d of %d) on %s\n", rank, commsize, host);
        /* Receive messages from all processes > 0 */
        for (int i = 1; i < commsize; i++) {
            MPI_Recv(msg, NELEMS(msg), MPI_CHAR, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, MPI_CHAR, &count);
            printf("Message from %d (tag %d, count %d): '%s'\n", status.MPI_SOURCE,status.MPI_TAG, count, msg);
        }
    }

    MPI_Finalize();
    return 0;
}
