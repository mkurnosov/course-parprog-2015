#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <mpi.h>

typedef struct {
    double x;
    double y; 
    double z;
    double f;
    int data[8];
} particle_t; 

int main(int argc, char **argv)
{
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int nparticles = 1000;
    particle_t *particles = malloc(sizeof(*particles) * nparticles);
        
    /* Create data type for message of type msg_t */
    MPI_Datatype types[5] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
    int blocklens[5] = {1, 1, 1, 1, 8}; 
    MPI_Aint displs[0];
    displs[0] = offsetof(particle_t, x); 
    displs[1] = offsetof(particle_t, y); 
    displs[2] = offsetof(particle_t, z); 
    displs[3] = offsetof(particle_t, f); 
    displs[4] = offsetof(particle_t, data); 
    MPI_Datatype parttype;
    
    MPI_Type_create_struct(5, blocklens, displs, types, &parttype);
    MPI_Type_commit(&parttype);
    
    /* Init particles */
    if (rank == 0) {
        // Random positions in simulation box
        for (int i = 0; i < nparticles; i++) {
            particles[i].x = rand() % 10000;
            particles[i].y = rand() % 10000;
            particles[i].z = rand() % 10000;
            particles[i].f = 0.0;
        }
    }
    MPI_Bcast(particles, nparticles, parttype, 0, MPI_COMM_WORLD);
        
    MPI_Type_free(&parttype);
    free(particles);
    MPI_Finalize( );
    return 0;
}
