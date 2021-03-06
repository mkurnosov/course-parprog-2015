/*
 * sgemm.cu:
 *
 */

#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

enum {
    N = 1024
};

__global__ void sgemm(const float *a, const float *b, float *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;  

    if (row < n && col < n) {
        float s = 0.0;
        for (int k = 0; k < n; k++)
            s += a[row * n + k] * b[k * n + col];
        c[row * n + col] = s;
    }
}

void sgemm_host(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0.0;
            for (int k = 0; k < n; k++)
                s += a[i * n + k] * b[k * n + j];
            c[i * n + j] = s;
        }            
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
    double tcpu, tgpu, tmem;
    cudaError_t err;
    
    /* Allocate memory on host */
    size_t size = sizeof(float) * N * N;
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Allocation error.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 2.0;
        h_B[i] = 3.0;
    }
    
    tcpu = -wtime();
    sgemm_host(h_A, h_B, h_C, N);
    tcpu += wtime();

    // Verify that the result vector is correct
    for (int i = 0; i < N * N; i++) {
        if (fabs(6.0 * N - h_C[i]) > 1e-5) {
            fprintf(stderr, "CPU results verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    
    /* Allocate vectors on device */
    float *d_A = NULL, *d_B = NULL,  *d_C = NULL;
    if (cudaMalloc((void **)&d_A, size) != cudaSuccess) {
        fprintf(stderr, "Allocation error\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&d_B, size) != cudaSuccess) {
        fprintf(stderr, "Allocation error\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&d_C, size) != cudaSuccess) {
        fprintf(stderr, "Allocation error\n");
        exit(EXIT_FAILURE);
    }

    /* Copy the host vectors to device */
    tmem = -wtime();
    if (cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Host to device copying failed\n");
        exit(EXIT_FAILURE);        
    }
    if (cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Host to device copying failed\n");
        exit(EXIT_FAILURE);        
    }
    tmem += wtime();
    
    /* Launch the kernel */
    int threadsPerBlockDim = 32;
    dim3 blockDim(threadsPerBlockDim, threadsPerBlockDim, 1);    
    int blocksPerGridDimX = ceilf(N / (float)threadsPerBlockDim);
    int blocksPerGridDimY = ceilf(N / (float)threadsPerBlockDim);
    dim3 gridDim(blocksPerGridDimX, blocksPerGridDimY, 1);
    printf("CUDA kernel launch with %d (%d %d) blocks of %d (%d %d) threads\n", 
           blocksPerGridDimX * blocksPerGridDimY, blocksPerGridDimX, blocksPerGridDimY,
           threadsPerBlockDim * threadsPerBlockDim, threadsPerBlockDim, threadsPerBlockDim);
    tgpu = -wtime();
    sgemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    tgpu += wtime();
    
    if ( (err = cudaGetLastError()) != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Copy the device vectors to host */
    tmem -= wtime();
    if (cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Device to host copying failed\n");
        exit(EXIT_FAILURE);
    }
    tmem += wtime();

    // Verify that the result vector is correct
    for (int i = 0; i < N * N; i++) {
        if (fabs(6.0 * N - h_C[i]) > 1e-5) {
            fprintf(stderr, "GPU results verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    
    printf("CPU version (sec.): %.6f\n", tcpu);
    printf("GPU version (sec.): %.6f\n", tgpu);
    printf("Memory ops. (sec.): %.6f\n", tmem);
    printf("Memory bw. (MiB/sec.): %.2f\n", ((3 * size) >> 20) / tmem);
    printf("Speedup: %.2f\n", tcpu / tgpu);
    printf("Speedup (with mem ops.): %.2f\n", tcpu / (tgpu + tmem));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaDeviceReset();
    return 0;
}

