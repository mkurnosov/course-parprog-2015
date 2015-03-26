/*
 * hist.c: Example of histogram calculation in OpenMP.
 *
 * (C) 2015 Mikhail Kurnosov <mkurnosov@gmail.com>
 */
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <limits.h>
#include <omp.h>

const uint64_t width = 32 * 1024;
const uint64_t height = 32 * 1024;

void *xmalloc(size_t size)
{
    void *p = malloc(size);
    if (p == NULL) {
        fprintf(stderr, "No enough memory\n");
        exit(EXIT_FAILURE);
    }
    return p;    
}

void hist_serial(uint8_t *pixels, int height, int width)
{
    uint64_t npixels = height * width;
    
    // Number of an occurrence of an each pixel in the image
    int *h = xmalloc(sizeof(*h) * 256);
    for (int i = 0; i < 256; i++)
        h[i] = 0;

    for (int i = 0; i < npixels; i++)
        h[pixels[i]]++;

    int mini, maxi;
    for (mini = 0; mini < 256 && h[mini] == 0; mini++);
    for (maxi = 255; maxi >= 0 && h[maxi] == 0; maxi--);   
    
    int q = 255 / (maxi - mini);
    for (int i = 0; i < npixels; i++)
        pixels[i] = (pixels[i] - mini) * q;

    free(h);
}

void hist_omp(uint8_t *pixels, int height, int width)
{
    uint64_t npixels = height * width;
    
    // Number of an occurrence of an each pixel in the image
    int *h = xmalloc(sizeof(*h) * 256);
    for (int i = 0; i < 256; i++)
        h[i] = 0;
          
    #pragma omp parallel
    {
        // Local histogram for each thread
        int *hloc = xmalloc(sizeof(*hloc) * 256);
        for (int i = 0; i < 256; i++)        
            hloc[i] = 0;
    
        #pragma omp for nowait
        for (int i = 0; i < npixels; i++)
            hloc[pixels[i]]++;
        
        #pragma omp critical
        {
            for (int i = 0; i < 256; i++)
                h[i] += hloc[i];
        }               
        free(hloc);
                
        #pragma omp barrier
        
        int mini, maxi;
        for (mini = 0; mini < 256 && h[mini] == 0; mini++);
        for (maxi = 255; maxi >=0 && h[maxi] == 0; maxi--);

        int q = 255 / (maxi - mini);
        #pragma omp for
        for (int i = 0; i < npixels; i++)
            pixels[i] = (pixels[i] - mini) * q;
    }
    
    free(h);
}

void hist_omp2(uint8_t *pixels, int height, int width)
{
    uint64_t npixels = height * width;
    
    // Number of an occurrence of an each pixel in the image
    int *h = xmalloc(sizeof(*h) * 256);
    for (int i = 0; i < 256; i++)
        h[i] = 0;

    int mini = 256, maxi = -1;
          
    #pragma omp parallel
    {
        int *hloc = xmalloc(sizeof(*hloc) * 256);
        for (int i = 0; i < 256; i++)        
            hloc[i] = 0;
    
        #pragma omp for nowait
        for (int i = 0; i < npixels; i++)
            hloc[pixels[i]]++;

        int mini_loc, maxi_loc;
        for (mini_loc = 0; mini_loc < 256 && hloc[mini_loc] == 0; mini_loc++);
        for (maxi_loc = 255; maxi_loc >= 0 && hloc[maxi_loc] == 0; maxi_loc--);
                
        #pragma omp critical
        {
            if (mini > mini_loc)
                mini = mini_loc;
            if (maxi < maxi_loc)
                maxi = maxi_loc;
        }
        
        int q = 255 / (maxi - mini);
        #pragma omp for
        for (int i = 0; i < npixels; i++)
            pixels[i] = (pixels[i] - mini) * q;

        free(hloc);
    }
    
    free(h);
}

int main(int argc, char *argv[])
{
    printf("Histogram (image %dx%d ~ %" PRIu64 " MiB)\n", height, width, height * width / (1 << 20));
    uint64_t npixels = width * height;
    uint8_t *pixels1, *pixels2;

    // Run serial version
    pixels1 = xmalloc(sizeof(*pixels1) * npixels);
    srand(0);
    for (int i = 0; i < npixels; i++)
        pixels1[i] = rand() % 256;
        //pixels1[i] = (i / width) * (i % width);

    double tser = omp_get_wtime();
    hist_serial(pixels1, height, width);
    tser = omp_get_wtime() - tser;
    printf("Elapsed time (serial): %.6f\n", tser);

    // Run parallel version
    pixels2 = xmalloc(sizeof(*pixels2) * npixels);
    srand(0);
    for (int i = 0; i < npixels; i++)
        pixels2[i] = rand() % 256;
        //pixels2[i] = (i / width) * (i % width);

    double tpar = omp_get_wtime();
    hist_omp(pixels2, height, width);
    tpar = omp_get_wtime() - tpar;
    printf("Elapsed time (parallel): %.6f\n", tpar);

    printf("Speedup: %.2f\n", tser / tpar);

    for (int i = 0; i < npixels; i++) {
        if (pixels1[i] != pixels2[i]) {
            printf("Verification failed: %i %d %d \n", i, pixels1[i], pixels2[i]);
            break;
        }
    }    
    free(pixels1);
    free(pixels2);    
   
    return 0;
}
