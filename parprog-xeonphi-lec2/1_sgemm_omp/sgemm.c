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
    N = 2000,   
    M = 2000,
    Q = 2000,
    NREPS = 10,
};
    
double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

/* Naive matrix multiplication C[n, q] = A[n, m] * B[m, q] */
void sgemm_host(float *a, float *b, float *c, int n, int m, int q)
{
    /* FP ops: 2 * n * q * m */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < q; j++) {
            float s = 0.0;
            for (int k = 0; k < m; k++)
                s += a[i * m + k] * b[k * q + j];
            c[i * q + j] = s;
		}
	}
}

/* Matrix multiplication C[n, q] = A[n, m] * B[m, q] */
void sgemm_host_opt(float *a, float *b, float *c, int n, int m, int q)
{
    /* Permute loops k and j for improving cache utilization */
    for (int i = 0; i < n * q; i++)
        c[i] = 0;
    /* FP ops: 2 * n * m * q */    
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {   
            for (int j = 0; j < q; j++)
                c[i * q + j] += a[i * m + k] * b[k * q + j];
		}
	}
}

/* Matrix multiplication C[n, q] = A[n, m] * B[m, q] */
void sgemm_host_omp(float *a, float *b, float *c, int n, int m, int q)
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

double run_host(const char *msg, void (*sgemm_fun)(float *, float *, float *, int, int, int))
{
    double gflop = 2.0 * N * Q * M * 1E-9;
    float *a, *b, *c;
    
    a = malloc(sizeof(*a) * N * M);
    b = malloc(sizeof(*b) * M * Q);
    c = malloc(sizeof(*c) * N * Q);
    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "No enough memory\n");
        exit(EXIT_FAILURE);
    }
    
    srand(0);
    //#pragma omp parallel for 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++)
            a[i * M + j] = rand() % 100;      
	}
    //#pragma omp parallel for 
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < Q; j++)
            b[i * Q + j] = rand() % 100;      
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
        
    free(c);
    free(b);
    free(a);        
    return tavg;
}

int main(int argc, char **argv)
{
    int omp_only = (argc > 1) ? 1 : 0;
    
    printf("SGEMM N = %d, M = %d, Q = %d\n", N, M, Q);
    if (!omp_only) {
        double t_host = run_host("Host serial", &sgemm_host);
        double t_host_opt = run_host("Host opt", &sgemm_host_opt);
        double t_host_omp = run_host("Host OMP", &sgemm_host_omp);
    
        printf("Speedup (host/host_opt): %.2f\n", t_host / t_host_opt);
        printf("Speedup (host_opt/host_OMP): %.2f\n", t_host_opt / t_host_omp);
    } else {
        char buf[256];
        sprintf(buf, "Host OMP %d", omp_get_max_threads());
        run_host(buf, &sgemm_host_omp); 
    }    
    return 0;
}
