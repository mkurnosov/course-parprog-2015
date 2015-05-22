/*
 * life-mpi-2d.c: MPI implementation of the "Game of Life" (4 neighbours rules).
 *
 * Simple 2D decompisiton of grid is implemented.
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
#define IND(i, j) ((i) * (local_cols + 2) + (j))

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

void simulate_life(cell_t *local_grid, cell_t *local_newgrid, int local_rows, int local_cols, 
                   int ticks, int px, int py)
{
    // 5 point stencil
    int states[2][5] = {
        {0, 1, 1, 0, 0}, /* New states for a dead cell */
        {0, 1, 1, 0, 0}  /* New states for an alive cell */
    };
        
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int ranky = rank / px;
    int rankx = rank % px;

    int top = (ranky > 0) ? (ranky - 1) * px + rankx : MPI_PROC_NULL;
    int bottom = (ranky < py - 1) ? (ranky + 1) * px + rankx : MPI_PROC_NULL;
    int left = (rankx > 0) ? ranky * px + (rankx - 1) : MPI_PROC_NULL;
    int right = (rankx < px - 1) ? ranky * px + (rankx + 1) : MPI_PROC_NULL;

    MPI_Datatype col;     
    MPI_Type_vector(local_rows, 1, local_cols + 2, MPI_UINT8_T, &col);  
    MPI_Type_commit(&col);
    
    MPI_Datatype row;        
    MPI_Type_contiguous(local_cols, MPI_UINT8_T, &row);
    MPI_Type_commit(&row);

    MPI_Request reqs[8];
    double tmpi = 0;
    
    // Simulate life
    for (int tick = 0; tick < ticks; tick++) {        
        tmpi -= MPI_Wtime();
        MPI_Isend(&local_grid[IND(1, 1)], 1, row, top, 0, MPI_COMM_WORLD, &reqs[0]);                 // top
        MPI_Isend(&local_grid[IND(local_rows, 1)], 1, row, bottom, 0, MPI_COMM_WORLD, &reqs[1]);     // bottom
        MPI_Isend(&local_grid[IND(1, 1)], 1, col, left, 0, MPI_COMM_WORLD, &reqs[2]);                // left
        MPI_Isend(&local_grid[IND(1, local_cols)], 1, col, right, 0, MPI_COMM_WORLD, &reqs[3]);      // right
        MPI_Irecv(&local_grid[IND(0, 1)], 1, row, top, 0, MPI_COMM_WORLD, &reqs[4]);                 // top
        MPI_Irecv(&local_grid[IND(local_rows + 1, 1)], 1, row, bottom, 0, MPI_COMM_WORLD, &reqs[5]); // bottom
        MPI_Irecv(&local_grid[IND(1, 0)], 1, col, left, 0, MPI_COMM_WORLD, &reqs[6]);                // left
        MPI_Irecv(&local_grid[IND(1, local_cols + 1)], 1, col, right, 0, MPI_COMM_WORLD, &reqs[7]);  // right
        MPI_Waitall(8, reqs, MPI_STATUS_IGNORE);
        tmpi += MPI_Wtime();
        
        // Update cells
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= local_cols; j++) {
                // Count sum of 4 neighbours
                int curr_state = local_grid[IND(i, j)];
                int sum = local_grid[IND(i - 1, j)] +
                          local_grid[IND(i + 1, j)] +
                          local_grid[IND(i, j - 1)] +
                          local_grid[IND(i, j + 1)];
                local_newgrid[IND(i, j)] = states[curr_state][sum];
            }
        }        
        cell_t *p = local_grid;
        local_grid = local_newgrid;
        local_newgrid = p;
    }
    MPI_Type_free(&row);
    MPI_Type_free(&col);
    printf("Process %2d mpi time: %.6f\n", rank, tmpi);
}

int main(int argc, char *argv[]) 
{
    int commsize, rank;
    MPI_Init(&argc, &argv);
    double ttotal = -MPI_Wtime();
    
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);   
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        
    int px, py, rows, cols, ticks;
    
    // Broadcast command line arguments
    if (rank == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s <px> <py> [<rows> <cols> <ticks>]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        px = atoi(argv[1]);
        py = atoi(argv[2]);
        rows = (argc > 3) ? atoi(argv[3]) : 100;
        cols = (argc > 4) ? atoi(argv[4]) : 100;
        ticks = (argc > 5) ? atoi(argv[5]) : 10;
        
        if (px * py != commsize) {
            fprintf(stderr, "Invalid values of <px> and <py>: %d * %d != %d\n", px, py, commsize);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (rows % py != 0 || cols % px != 0) {
            fprintf(stderr, "px and py must be devisors of rows and cols\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        if (rows < py) {
            fprintf(stderr, "Number of rows %d less then number of py processes %d\n", rows, py);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (cols < px) {
            fprintf(stderr, "Number of cols %d less then number of px processes %d\n", cols, px);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        
        int args[5] = {px, py, rows, cols, ticks};
        MPI_Bcast(&args, NELEMS(args), MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        int args[5];
        MPI_Bcast(&args, NELEMS(args), MPI_INT, 0, MPI_COMM_WORLD);
        px = args[0]; 
        py = args[1];
        rows = args[2];
        cols = args[3];
        ticks = args[4];
    }    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Allocate memory for 2D subgrids with halo cells [0..local_rows + 1][0..local_cols + 1]
    int local_rows = rows / py;
    int local_cols = cols / px;
    cell_t *local_grid = xcalloc((local_rows + 2) * (local_cols + 2), sizeof(*local_grid));
    cell_t *local_newgrid = xcalloc((local_rows + 2) * (local_cols + 2), sizeof(*local_newgrid));

    // Fill 2D subgrid
    srand(rank);
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= local_cols; j++)
            local_grid[IND(i, j)] = rand() % 10 > 0 ? 0 : 1;
    }
    
    simulate_life(local_grid, local_newgrid, local_rows, local_cols, ticks, px, py);   
    
    free(local_newgrid);
    free(local_grid);

    ttotal += MPI_Wtime();
    printf("Process %2d total time: %.6f\n", rank, ttotal);
        
    MPI_Finalize();    
    return 0;
}
