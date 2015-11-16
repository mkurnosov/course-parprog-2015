#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

const float eps = 0.0001f;
const float dt = 0.01f;
const int N = 128 * 1024;

#define coord float3

__global__ void integrate(coord *new_p, coord *new_v, coord *p, coord *v, 
                          int n, float dt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;    
    if (index >= n)
        return;
    
    coord body_pos = p[index];      
    coord body_vel = v[index];      
    coord f;
    f.x = 0;
    f.y = 0;
    f.z = 0;

    for (int i = 0; i < n; i++) {
        coord pi = p[i];            
        coord r;
        // Vector from p[i] to body
        r.x = pi.x - body_pos.x;    
        r.y = pi.y - body_pos.y;    
        r.z = pi.z - body_pos.z;    
        
        float invDist = 1.0 / sqrtf(r.x * r.x + r.y * r.y + r.z * r.z + eps * eps);
        float s = invDist * invDist * invDist;
        // Add force of body i
        f.x += r.x * s;
        f.y += r.y * s;
        f.z += r.z * s;
    }
    
    // Correct velocity
    body_vel.x += f.x * dt;
    body_vel.y += f.y * dt;
    body_vel.z += f.z * dt;
    body_pos.x += body_vel.x * dt;
    body_pos.y += body_vel.y * dt;
    body_pos.z += body_vel.z * dt;
    
    new_p[index] = body_pos;
    new_v[index] = body_vel;
}

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

void init_rand(coord *v, int n)
{
    for (int i = 0; i < n; i++) {
        v[i].x = rand() / (float)RAND_MAX - 0.5f;
        v[i].y = rand() / (float)RAND_MAX - 0.5f;
        v[i].z = rand() / (float)RAND_MAX - 0.5f;
    }
}

int main()
{
    double tgpu = 0, tmem = 0;
    
    size_t size = sizeof(coord) * N;
    coord *p = (coord *)malloc(size);
    coord *v = (coord *)malloc(size);
    coord *d_p[2] = {NULL, NULL};
    coord *d_v[2] = {NULL, NULL};
    
    init_rand(p, N);
    init_rand(v, N);
    
    tmem = -wtime();
    cudaMalloc((void **)&d_p[0], size);
    cudaMalloc((void **)&d_p[1], size);
    cudaMalloc((void **)&d_v[0], size);
    cudaMalloc((void **)&d_v[1], size);
    cudaMemcpy(d_p[0], p, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v[0], v, size, cudaMemcpyHostToDevice);
    tmem += wtime();
        
    tgpu = -wtime();
    int threadsPerBlock = 1024;
    dim3 block(threadsPerBlock);
    dim3 grid((N + threadsPerBlock - 1) / threadsPerBlock);    
    int index = 0;
    for (int i = 0; i < 2; i++, index ^= 1) {
        integrate<<<grid, block>>>(d_p[index ^ 1], d_v[index ^ 1], d_p[index], d_v[index], N, dt);
    }
    cudaDeviceSynchronize();
    tgpu += wtime();
    
    tmem -= wtime();
    cudaMemcpy(p, d_p[index], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v[index], size, cudaMemcpyDeviceToHost);
    tmem += wtime();
    
    /*
    for (int i = 0; i < N; i++) {
        printf("%4d: %f %f %f %f %f %f\n", i, p[i].x, p[i].y, p[i].z, v[i].x, v[i].y, v[i].z);
    }
    */
    
    printf("sizeof(coord) = %d\n", sizeof(coord));
    printf("GPU version (sec.): %.6f\n", tgpu);
    printf("Memory ops. (sec.): %.6f\n", tmem);
    printf(" Total time (sec.): %.6f\n", tgpu + tmem);

    cudaFree(d_p[0]);
    cudaFree(d_p[1]);
    cudaFree(d_v[0]);
    cudaFree(d_v[1]);
    free(p);
    free(v);
    cudaDeviceReset();
    return 0;
}

