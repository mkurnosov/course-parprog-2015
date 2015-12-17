#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/time.h>

#define IND(i, j) ((i) * (N + 2) + (j))

enum {
    N = 1024,
    ITERS_MAX = 1 << 10
};

typedef uint8_t cell_t;

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

int main(int argc, char* argv[])
{
    // Grid with periodic boundary conditions (ghost cells)
    size_t ncells = (N + 2) * (N + 2);
    size_t size = sizeof(cell_t) * ncells;
    cell_t *grid = malloc(size);
    cell_t *newgrid = malloc(size);
 
    // Initial population
    srand(0);
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            grid[IND(i, j)] = rand() % 2;
    
    double t = wtime();
    int iter;
    for (iter = 0; iter < ITERS_MAX; iter++) {
        // Copy ghost columns
        for (int i = 1; i <= N; i++) {
            grid[IND(i, 0)] = grid[IND(i, N)];      // left ghost column
            grid[IND(i, N + 1)] = grid[IND(i, 1)];  // right ghost column
        }
        // Copy ghost rows
        for (int i = 0; i <= N + 1; i++) {
            grid[IND(0, i)] = grid[IND(N, i)];      // top ghost row
            grid[IND(N + 1, i)] = grid[IND(1, i)];  // bottom ghost row
        }
 
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                int nneibs = grid[IND(i + 1, j)] + grid[IND(i - 1, j)] +
                             grid[IND(i, j + 1)] + grid[IND(i, j - 1)] +
                             grid[IND(i + 1, j + 1)] + grid[IND(i - 1, j - 1)] +
                             grid[IND(i - 1, j + 1)] + grid[IND(i + 1, j - 1)];
 
                cell_t state = grid[IND(i, j)];
                cell_t newstate = state;
                if (state == 1 && nneibs < 2)
                    newstate = 0;
                else if (state == 1 && (nneibs == 2 || nneibs == 3))
                    newstate = 1;
                else if (state == 1 && nneibs > 3)
                    newstate = 0;
                else if (state == 0 && nneibs == 3)
                    newstate = 1;
                newgrid[IND(i, j)] = newstate;
            }
        }
 
        cell_t *p = grid;
        grid = newgrid;
        newgrid = p;
    }
    t = wtime() - t;
    
    size_t total = 0;
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++)
            total += grid[IND(i, j)];
    }
    printf("Game of Life: N = %d, iterations = %d\n", N, iter);
    printf("Total alive cells: %lu\n", total);
    printf("Iters per sec.: %.2f\n", iter / t);
    printf("Total time (sec.): %.6f\n", t);
 
    free(grid);
    free(newgrid);
    return 0;
}
