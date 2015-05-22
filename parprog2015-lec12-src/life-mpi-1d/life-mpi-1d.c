/*
 * life-mpi-1d.c: MPI implementation of the "Game of Life" (4 neighbours rules).
 *
 * Simple 1D decompisiton of grid is implemented.
 *
 * (C) Mikhail Kurnosov, 2015
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

#define NELEMS(x) (sizeof((x)) / sizeof((x)[0]))
#define IND(i, j) ((i) * cols + (j))

#define ALIVE_CHAR 'X'
#define DEAD_CHAR  '.'

typedef uint8_t cell_t;

void *xcalloc(size_t nmemb, size_t size)
{
    void *p = calloc(nmemb, size);
    if (p == NULL) {
        fprintf(stderr, "No enough memory\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return p;    
}

void simulate_life(cell_t *local_grid, cell_t *local_newgrid, int local_rows, int cols, int ticks)
{
    // 5 point stencil
    int states[2][5] = {
        {0, 1, 1, 0, 0}, /* New states for a dead cell */
        {0, 1, 1, 0, 0}  /* New states for an alive cell */
    };
        
    int rank, commsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    
    int top = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int bottom = (rank < commsize - 1) ? rank + 1 : MPI_PROC_NULL;

    MPI_Datatype row;
    MPI_Type_contiguous(cols, MPI_UINT8_T, &row);
    MPI_Type_commit(&row);

    MPI_Request reqs[4];
    double tmpi = 0;
    
    // Simulate life
    for (int tick = 0; tick < ticks; tick++) {
        tmpi -= MPI_Wtime();
        MPI_Isend(&local_grid[IND(1, 0)], 1, row, top, 0, MPI_COMM_WORLD, &reqs[0]);                 // top
        MPI_Isend(&local_grid[IND(local_rows, 0)], 1, row, bottom, 0, MPI_COMM_WORLD, &reqs[1]);     // bottom
        MPI_Irecv(&local_grid[IND(0, 0)], 1, row, top, 0, MPI_COMM_WORLD, &reqs[2]);                 // top
        MPI_Irecv(&local_grid[IND(local_rows + 1, 0)], 1, row, bottom, 0, MPI_COMM_WORLD, &reqs[3]); // bottom
        MPI_Waitall(4, reqs, MPI_STATUS_IGNORE);
        tmpi += MPI_Wtime();
        
        // Update cells
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Count sum of 4 neighbours
                int curr_state = local_grid[IND(i, j)];
                int sum = local_grid[IND(i - 1, j)] +
                          local_grid[IND(i + 1, j)];
                if (j > 0)
                    sum += local_grid[IND(i, j - 1)];
                if (j < cols - 1)                    
                    sum += local_grid[IND(i, j + 1)];
                local_newgrid[IND(i, j)] = states[curr_state][sum];
            }
        }        
        cell_t *p = local_grid;
        local_grid = local_newgrid;
        local_newgrid = p;
    }
    MPI_Type_free(&row);
    printf("Process %2d mpi time: %.6f\n", rank, tmpi);
}

int main(int argc, char *argv[]) 
{
    int commsize, rank;
    MPI_Init(&argc, &argv);
    double ttotal = -MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);   
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        
    int rows, cols, ticks;
    
    // Broadcast command line arguments
    if (rank == 0) {
        rows = (argc > 1) ? atoi(argv[1]) : 100;
        cols = (argc > 2) ? atoi(argv[2]) : 100;
        ticks = (argc > 3) ? atoi(argv[3]) : 10;
        
        if (rows % commsize != 0) {
            fprintf(stderr, "commsize must be devisor of rows\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        if (rows < commsize) {
            fprintf(stderr, "Number of rows %d less then number of processes %d\n", rows, commsize);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        
        int args[3] = {rows, cols, ticks};
        MPI_Bcast(&args, NELEMS(args), MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        int args[3];
        MPI_Bcast(&args, NELEMS(args), MPI_INT, 0, MPI_COMM_WORLD);
        rows = args[0];
        cols = args[1];
        ticks = args[2];
    }    
   
    // Allocate memory for 1D subgrids with halo cells [0..local_rows + 1][0..cols - 1]
    int local_rows = rows / commsize;
    cell_t *local_grid = xcalloc((local_rows + 2) * cols, sizeof(*local_grid));
    cell_t *local_newgrid = xcalloc((local_rows + 2) * cols, sizeof(*local_newgrid));

    // Fill 1D subgrid
    srand(rank);
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 0; j < cols; j++)
            local_grid[IND(i, j)] = rand() % 10 > 0 ? 0 : 1;
    }
    
    simulate_life(local_grid, local_newgrid, local_rows, cols, ticks);
    
    free(local_newgrid);
    free(local_grid);

    ttotal += MPI_Wtime();
    printf("Process %2d total time: %.6f\n", rank, ttotal);

    MPI_Finalize();
    
    return 0;
}
