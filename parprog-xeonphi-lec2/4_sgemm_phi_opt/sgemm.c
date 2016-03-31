#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

/* 
 * C = [n, q] = A[n, m] * B[m, q] 
 */

enum {
    N = 10000,   
    M = 10000,
    Q = 10000,
    NREPS = 3,
};
    
double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

/* Matrix multiplication C[n, q] = A[n, m] * B[m, q] */
void sgemm_phi(float *a, float *b, float *c, int n, int m, int q)
{
    #pragma offload target(mic) in(a:length(n * m)) in(b:length(m * q)) out(c:length(n * q))
    {
        #pragma omp parallel
        {
            int k = 0;
            #pragma omp for
            for (int i = 0; i < n; i++)
                for (int j = 0; j < q; j++)
                    c[k++] = 0.0;

            #pragma omp for
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < m; k++) {
                    for (int j = 0; j < q; j++)
                        c[i * q + j] += a[i * m + k] * b[k * q + j];
        		}
	        }
    	}
    }    	
}

double run_phi(const char *msg, void (*sgemm_fun)(float *, float *, float *, int, int, int))
{
    double gflop = 2.0 * N * Q * M * 1E-9;
    float *a, *b, *c;
    
    a = _mm_malloc(sizeof(*a) * N * M, 4096);
    b = _mm_malloc(sizeof(*b) * M * Q, 4096);
    c = _mm_malloc(sizeof(*c) * N * Q, 4096);
    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "No enough memory\n");
        exit(EXIT_FAILURE);
    }
    
    srand(0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++)
            a[i * M + j] = rand() % 100;      // 1.0;
	}
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < Q; j++)
            b[i * Q + j] = rand() % 100;      // 2.0;
	}
    
    /* Warmup */
    double twarmup = wtime();
    sgemm_fun(a, b, c, N, M, Q);
    twarmup = wtime() - twarmup;
    
    /* Measures */
    double tavg = 0.0;
    double tmin = 1E6;
    double tmax = 0.0;
    
    for (int i = 0; i < NREPS; i++) {
        double t = wtime();
        sgemm_fun(a, b, c, N, M, Q);
        t = wtime() - t;
        tavg += t;
        tmin = (tmin > t) ? t : tmin;
        tmax = (tmax < t) ? t : tmax;
    }   
    tavg /= NREPS;
    printf("%s (%d runs): perf %.2f GFLOPS; time: tavg %.6f, tmin %.6f, tmax %.6f, twarmup %.6f\n", 
           msg, NREPS, gflop / tavg, tavg, tmin, tmax, twarmup);
        
    _mm_free(c);
    _mm_free(b);
    _mm_free(a);        
    return tavg;
}

int main(int argc, char **argv)
{
    printf("SGEMM N = %d, M = %d, Q = %d\n", N, M, Q);
    char buf[256];
    sprintf(buf, "Phi OMP %s", getenv("MIC_OMP_NUM_THREADS"));
    run_phi(buf, &sgemm_phi); 
    
    return 0;
}
