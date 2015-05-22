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
                   int ticks, int px, int py, int rankx, int ranky, MPI_Comm cartcomm)
{
    // 5 point stencil
    int states[2][5] = {
        {0, 1, 1, 0, 0}, /* New states for a dead cell */
        {0, 1, 1, 0, 0}  /* New states for an alive cell */
    };
    
    int left, right, top, bottom;
    MPI_Cart_shift(cartcomm, 0, 1, &left, &right);
    MPI_Cart_shift(cartcomm, 1, 1, &top, &bottom);
    
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
        MPI_Irecv(&local_grid[IND(0, 1)], 1, row, top, 0, cartcomm, &reqs[4]);                 // top
        MPI_Irecv(&local_grid[IND(local_rows + 1, 1)], 1, row, bottom, 0, cartcomm, &reqs[5]); // bottom
        MPI_Irecv(&local_grid[IND(1, 0)], 1, col, left, 0, cartcomm, &reqs[6]);                // left
        MPI_Irecv(&local_grid[IND(1, local_cols + 1)], 1, col, right, 0, cartcomm, &reqs[7]);  // right
        MPI_Isend(&local_grid[IND(1, 1)], 1, row, top, 0, cartcomm, &reqs[0]);                 // top
        MPI_Isend(&local_grid[IND(local_rows, 1)], 1, row, bottom, 0, cartcomm, &reqs[1]);     // bottom
        MPI_Isend(&local_grid[IND(1, 1)], 1, col, left, 0, cartcomm, &reqs[2]);                // left
        MPI_Isend(&local_grid[IND(1, local_cols)], 1, col, right, 0, cartcomm, &reqs[3]);      // right
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
    printf("Process (%2d, %2d) mpi time: %.6f\n", ranky, rankx, tmpi);
}

int main(int argc, char *argv[]) 
{
    int commsize, rank;
    MPI_Init(&argc, &argv);
    double ttotal = -MPI_Wtime();
    
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);   
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       
        
    // 2D grid of processes
    MPI_Comm cartcomm;
    int dims[2] = {0, 0}, periodic[2] = {0, 0};
    MPI_Dims_create(commsize, 2, dims);
    int px = dims[0];
    int py = dims[1];    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 1, &cartcomm);    
    int coords[2];
    MPI_Cart_coords(cartcomm, rank, 2, coords);
    int rankx = coords[0];
    int ranky = coords[1];
    int namelen;
    char procname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(procname, &namelen);
    
    printf("Process %2d (%2d %2d) on %s\n", rank, ranky, rankx, procname);
        
    int rows, cols, ticks;
    
    // Broadcast command line arguments
    if (rank == 0) {
        rows = (argc > 1) ? atoi(argv[1]) : 100;
        cols = (argc > 2) ? atoi(argv[2]) : 100;
        ticks = (argc > 3) ? atoi(argv[3]) : 10;
        
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
        
        int args[3] = {rows, cols, ticks};
        MPI_Bcast(&args, NELEMS(args), MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        int args[3];
        MPI_Bcast(&args, NELEMS(args), MPI_INT, 0, MPI_COMM_WORLD);
        rows = args[0];
        cols = args[1];
        ticks = args[2];
    }    
    
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
    
    simulate_life(local_grid, local_newgrid, local_rows, local_cols, ticks, px, py, rankx, ranky, cartcomm);
    
    free(local_newgrid);
    free(local_grid);

    ttotal += MPI_Wtime();
    printf("Process %2d total time: %.6f\n", rank, ttotal);
        
    MPI_Finalize();    
    return 0;
}
