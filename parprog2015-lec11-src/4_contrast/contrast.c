#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <mpi.h>

void *xmalloc(size_t size)
{
    void *p = malloc(size);
    if (p == NULL) {
        fprintf(stderr, "No enough memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return p;    
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);   
    double mpitime = 0.0, totaltime = MPI_Wtime();
    int rank, commsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    
    // Load image
    int width, height, npixels, npixels_per_process;
    uint8_t *pixels = NULL;    
    if (rank == 0) {
        // 15360 x 8640: 16K Digital Cinema (UHDTV) ~ 127 MiB
        width = 15360;
        height = 8640;
        npixels = width * height;
        pixels = xmalloc(sizeof(*pixels) * npixels);
        for (int i = 0; i < npixels; i++)
            pixels[i] = rand() % 255;
    }
    
    mpitime -= MPI_Wtime();
    MPI_Bcast(&npixels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    mpitime += MPI_Wtime();
    npixels_per_process = npixels / commsize;
    
    uint8_t *rbuf = xmalloc(sizeof(*rbuf) * npixels_per_process);

    /* Send a part of image to each process */
    mpitime -= MPI_Wtime();
    MPI_Scatter(pixels, npixels_per_process, MPI_UINT8_T, rbuf, npixels_per_process, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    mpitime += MPI_Wtime();

    /* Take the sum of the squares of the partial image */
    int sum_local = 0;
    for (int i = 0; i < npixels_per_process; i++)
        sum_local += rbuf[i] * rbuf[i];

    /* Calculate global sum of the squares */
    int sum = 0;
    mpitime -= MPI_Wtime();
    //MPI_Reduce(&sum_local, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&sum_local, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    mpitime += MPI_Wtime();

    double rms;
    //if (rank == 0)
    rms = sqrt((double)sum / (double)npixels);
    
    //mpitime -= MPI_Wtime();
    //MPI_Bcast(&rms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //mpitime += MPI_Wtime();

    /* Contrast operation on subimage */
    for (int i = 0; i < npixels_per_process; i++) {
        int pixel = 2 * rbuf[i] - rms;
        if (pixel < 0)
            rbuf[i] = 0;
        else if (pixel > 255)
            rbuf[i] = 255;
        else
            rbuf[i] = pixel;
    }
    mpitime -= MPI_Wtime();
    MPI_Gather(rbuf, npixels_per_process, MPI_UINT8_T, pixels, npixels_per_process, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    mpitime += MPI_Wtime();

    if (rank == 0) {
        FILE *fout = fopen("image-out.dat", "w");
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                fprintf(fout, "%4d", pixels[i * width + j]);
            }
            fprintf(fout, "\n");
        }
        fclose(fout);
    }
    
    free(rbuf);
    if (rank == 0)
        free(pixels);

    totaltime = MPI_Wtime() - totaltime;
    printf("Process %d: totaltime=%.6f;  comptime=%.6f;  mpitime=%.6f;  overhead=%.2f\n", 
           rank, totaltime, totaltime - mpitime, mpitime, (double)mpitime / ((double)totaltime - mpitime));
    MPI_Finalize();
    return 0;
}
