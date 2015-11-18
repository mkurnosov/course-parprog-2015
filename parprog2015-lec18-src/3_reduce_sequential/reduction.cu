#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

const int block_size = 1024;
const int n = 4 * (1 << 20);

void reduce_cpu(int *v, int n, int *sum)
{
    /*
    int s = 0.0;
    for (int i = 0; i < n; i++)
        s += v[i];        
    *sum = s;    
    */

    // Kahan's summation algorithm
    int s = v[0];
    int c = (int)0.0;

    for (int i = 1; i < n; i++) {
        int y = v[i] - c;
        int t = s + y;
        c = (t - s) - y;
        s = t;
    }
    *sum = s;
}

__global__ void reduce_per_block(int *v, int n, int *per_block_sum)
{
    __shared__ int sdata[block_size];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
        
    if (i < n) {
        sdata[tid] = v[i];
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            per_block_sum[blockIdx.x] = sdata[0];
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
    
    size_t size = sizeof(int) * n;
    int *v = (int *)malloc(size);

    srand(0);
    for (size_t i = 0; i < n; i++)
        v[i] = i + 1;

    int sum;
    tcpu = -wtime();
    reduce_cpu(v, n, &sum);
    tcpu += wtime();
    
    /* Allocate on device */
    int threads_per_block = block_size;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    int *dv;    
    int *per_block_sum;
    int *sums = (int *)malloc(sizeof(int) * blocks);
    tmem = -wtime();
    cudaMalloc((void **)&per_block_sum, sizeof(int) * blocks);
    cudaMalloc((void **)&dv, size);
    cudaMemcpy(dv, v, size, cudaMemcpyHostToDevice);
    tmem += wtime();
        
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocks, threads_per_block);
    fflush(stdout);
    
    /* Compute per block sum */
    tgpu = -wtime();
    reduce_per_block<<<blocks, threads_per_block>>>(dv, n, per_block_sum);
    cudaDeviceSynchronize();
    tgpu += wtime();    

    tmem = -wtime();
    cudaMemcpy(sums, per_block_sum, sizeof(int) * blocks, cudaMemcpyDeviceToHost);
    tmem += wtime();

    /* Compute block sum */
    tgpu -= wtime();
    int sum_gpu = 0;
    for (int i = 0; i < blocks; i++)
        sum_gpu += sums[i];
    tgpu += wtime();
    
    printf("Sum (CPU) = %d\n", sum);
    printf("Sum (GPU) = %d\n", sum_gpu);

    printf("CPU version (sec.): %.6f\n", tcpu);
    printf("GPU version (sec.): %.6f\n", tgpu);
    printf("GPU bandwidth (GiB/s): %.2f\n", 1.0e-9 * size / (tgpu + tmem));
    printf("Speedup: %.2f\n", tcpu / tgpu);
    printf("Speedup (with mem ops.): %.2f\n", tcpu / (tgpu + tmem));

    cudaFree(per_block_sum);
    cudaFree(dv);
    free(sums);
    free(v);
    cudaDeviceReset();
    return 0;
}

