#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define IND(i, j) ((i) * (N + 2) + (j))

enum {
    N = 1024,
    ITERS_MAX = 1,
    BLOCK_1D_SIZE = 1024,
    BLOCK_2D_SIZE = 32
};

typedef uint8_t cell_t;

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

__global__ void copy_ghost_rows(cell_t *grid, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n + 1) {
        // Bottom ghost row: [N + 1][0..N + 1] <== [1][0..N + 1]
        grid[IND(N + 1, i)] = grid[IND(1, i)];
        // Top ghost row: [0][0..N + 1] <== [N][0..N + 1]
        grid[IND(0, i)] = grid[IND(N, i)];
    }
}

__global__ void copy_ghost_cols(cell_t *grid, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i <= n) {
        // Right ghost column: [1..N][N + 1] <== [1..N][1]
        grid[IND(i, N + 1)] = grid[IND(i, 1)];
        // Left ghost column: [1..N][1] <== [1..N][N]
        grid[IND(i, 0)] = grid[IND(i, N)];
    }
}

__constant__ int states[2][9] = {
    {0, 0, 0, 1, 0, 0, 0, 0, 0},  /* New states for a dead cell */
    {0, 0, 1, 1, 0, 0, 0, 0, 0}   /* New states for an alive cell */
};

__global__ void update_cells(cell_t *grid, cell_t *newgrid, int n)
{
    int iy = blockIdx.y * (blockDim.y - 2) + threadIdx.y;
    int ix = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
    int i = threadIdx.y;
    int j = threadIdx.x;
    
    // Copy cells of the block into shared memory
    __shared__ cell_t s_grid[BLOCK_2D_SIZE][BLOCK_2D_SIZE];
    if (ix <= n + 1 && iy <= n + 1)
        s_grid[i][j] = grid[IND(iy, ix)];    
    __syncthreads();
    
    if (ix <= n && iy <= n) {        
        if (i > 0 && i != blockDim.y - 1 && j > 0 && j != blockDim.x - 1) {
            int nneibs = s_grid[i + 1][j] + s_grid[i - 1][j] + s_grid[i][j + 1] + s_grid[i][j - 1] + 
                         s_grid[i + 1][j + 1] + s_grid[i - 1][j - 1] + 
                         s_grid[i - 1][j + 1] + s_grid[i + 1][j - 1];
            
            cell_t state = s_grid[i][j];
            newgrid[IND(iy, ix)] = states[state][nneibs];
        }
    }
}

int main(int argc, char* argv[])
{
    // Grid with periodic boundary conditions (ghost cells)
    size_t ncells = (N + 2) * (N + 2);
    size_t size = sizeof(cell_t) * ncells;
    cell_t *grid = (cell_t *)malloc(size);
 
    // Initial population
    srand(0);
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            grid[IND(i, j)] = rand() % 2;

    cell_t *d_grid, *d_newgrid;
    double tmem = -wtime();
    cudaMalloc((void **)&d_grid, size);
    cudaMalloc((void **)&d_newgrid, size);
    cudaMemcpy(d_grid, grid, size, cudaMemcpyHostToDevice);
    tmem += wtime();
    
    // 1d grids for copying ghost cells
    dim3 block(BLOCK_1D_SIZE, 1, 1);
    dim3 cols_grid((N + block.x - 1) / block.x, 1, 1);
    dim3 rows_grid((N + 2 + block.x - 1) / block.x, 1, 1);
    
    // 2d grid for updating cells: one thread per cell
    dim3 block2d(BLOCK_2D_SIZE, BLOCK_2D_SIZE, 1);    
    // Boundary threads in block only for loading ghost data 
    int nblocks = ceilf(N / (BLOCK_2D_SIZE - 2));
    dim3 grid2d(nblocks, nblocks, 1);
    
    double t = wtime();
    int iter = 0;
    for (iter = 0; iter < ITERS_MAX; iter++) {
        // Copy ghost cells
        copy_ghost_cols<<<cols_grid, block>>>(d_grid, N);
        copy_ghost_rows<<<rows_grid, block>>>(d_grid, N);

        // Update cells
        update_cells<<<grid2d, block2d>>>(d_grid, d_newgrid, N);
        
        // Swap grids
        cell_t *p = d_grid;
        d_grid = d_newgrid;
        d_newgrid = p;
    }
    cudaDeviceSynchronize();
    t = wtime() - t;
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    tmem -= wtime();
    cudaMemcpy(grid, d_grid, size, cudaMemcpyDeviceToHost);
    tmem += wtime();
    
    /*
    for (int i = 0; i < N + 2; i++) {
        for (int j = 0; j < N + 2; j++)
            printf("%1d ", grid[IND(i, j)]);
        printf("\n");        
    }
    */
    size_t total = 0;
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++)
            total += grid[IND(i, j)];
    }
    printf("Game of Life: N = %d, iterations = %d\n", N, iter);
    printf("Total alive cells: %lu\n", total);
    printf("Iterations time (sec.): %.6f\n", t);
    printf("GPU memory ops. time (sec.): %.6f\n", tmem);
    printf("Iters per sec.: %.2f\n", iter / t);
    printf("Total time (sec.): %.6f\n", t + tmem);

    free(grid);    
    cudaFree(d_grid);
    cudaFree(d_newgrid);    
    return 0;
}
