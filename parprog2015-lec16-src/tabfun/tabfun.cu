#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

const float step = 0.001;

enum {
    BLOCK_SIZE = 32,
    N = 1024
};

void tabfun_host(float *tab, float step, int n)
{
    for (int i = 0; i < n; i++) {
        float x = step * i;
        tab[i] = sinf(sqrtf(x));
    }
}

__global__ void tabfun(float *tab, float step, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;    
    if (index < n) {
        float x = step * index;
        tab[index] = sinf(sqrtf(x));
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
    
    /* Allocate memory on host */
    size_t size = sizeof(float) * N;
    float *hT = (float *)malloc(size);
    float *hRes = (float *)malloc(size);
    if (hT == NULL || hRes == NULL) {
        fprintf(stderr, "Allocation error.\n");
        exit(EXIT_FAILURE);
    }
    
    tcpu = -wtime();
    tabfun_host(hT, step, N);
    tcpu += wtime();

    /* Allocate vectors on device */
    float *dT = NULL;
    if (cudaMalloc((void **)&dT, size) != cudaSuccess) {
        fprintf(stderr, "Allocation error\n");
        exit(EXIT_FAILURE);
    }

    /* Launch the kernel */
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", 
           blocksPerGrid, threadsPerBlock);
    tgpu = -wtime();
    tabfun<<<blocksPerGrid, threadsPerBlock>>>(dT, step, N);
    cudaDeviceSynchronize();
    tgpu += wtime();
    
    if ( (err = cudaGetLastError()) != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Copy the device vectors to host */
    tmem -= wtime();
    if (cudaMemcpy(hRes, dT, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Device to host copying failed\n");
        exit(EXIT_FAILURE);
    }
    tmem += wtime();

    // Verify that the result vector is correct
    for (int i = 0; i < N; i++) {
        float d = fabs(hT[i] - hRes[i]);
        printf("%d: %f\n", i, d);
    }
    
    printf("CPU version (sec.): %.6f\n", tcpu);
    printf("GPU version (sec.): %.6f\n", tgpu);
    printf("Memory ops. (sec.): %.6f\n", tmem);
    printf("Speedup: %.2f\n", tcpu / tgpu);
    printf("Speedup (with mem ops.): %.2f\n", tcpu / (tgpu + tmem));

    cudaFree(dT);
    free(hT);
    free(hRes);
    cudaDeviceReset();
    return 0;
}

