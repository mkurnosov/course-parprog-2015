#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/time.h>
#include <cuda_runtime.h>

const int width = 10 * 1024;
const int height = 10 * 1024;

void hist_host(uint8_t *image, int width, int height, int *hist)
{
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            hist[image[i * width + j]]++;
}

__global__ void hist_gpu(uint8_t *image, int width, int height, int *hist)
{
    size_t size = width * height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        hist[image[i]]++;
    }
}

__global__ void hist_gpu_atomic(uint8_t *image, int width, int height, int *hist)
{
    size_t size = width * height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < size) {
        atomicAdd(&hist[image[i]], 1);
        i += stride;
    }
}

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

int main()
{
    double tcpu = 0, tgpu = 0, tmem = 0;
    cudaError_t err;
    
    size_t size = sizeof(uint8_t) * width * height;
    uint8_t *image = (uint8_t *)malloc(size);
    if (image == NULL) {
        fprintf(stderr, "Allocation error.\n");
        exit(EXIT_FAILURE);
    }
    srand(0);
    for (size_t i = 0; i < size; i++)
        image[i] = (rand() / (double)RAND_MAX) * 255;

    int hist[256];
    memset(hist, 0, sizeof(*hist) * 256);
    
    tcpu = -wtime();
    hist_host(image, width, height, hist);
    tcpu += wtime();

    double sum = 0;
    for (int i = 0; i < 256; i++)
        sum += hist[i];    
    printf("Sum (CPU) = %f\n", sum);
            
    /* Allocate on device */
    int *d_hist = NULL;
    cudaMalloc((void **)&d_hist, sizeof(*d_hist) * 256);
    cudaMemset(d_hist, 0, sizeof(*d_hist) * 256);

    uint8_t *d_image = NULL;
    cudaMalloc((void **)&d_image, size);
    cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);
    
    /* Launch the kernel */
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    tgpu = -wtime();
    hist_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height, d_hist);
    cudaDeviceSynchronize();
    tgpu += wtime();    
    if ( (err = cudaGetLastError()) != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    tmem -= wtime();
    cudaMemcpy(hist, d_hist, sizeof(*d_hist) * 256, cudaMemcpyDeviceToHost);
    tmem += wtime();

    sum = 0;
    for (int i = 0; i < 256; i++)
        sum += hist[i];    
    printf("Sum (GPU) = %f\n", sum);

    printf("CPU version (sec.): %.6f\n", tcpu);
    printf("GPU version (sec.): %.6f\n", tgpu);
    printf("Memory ops. (sec.): %.6f\n", tmem);
    printf("Speedup: %.2f\n", tcpu / tgpu);
    printf("Speedup (with mem ops.): %.2f\n", tcpu / (tgpu + tmem));

    cudaFree(d_image);
    cudaFree(d_hist);
    free(image);
    cudaDeviceReset();
    return 0;
}

