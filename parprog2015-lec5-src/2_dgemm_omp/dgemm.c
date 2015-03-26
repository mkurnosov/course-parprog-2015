/*
 * dgemm.c: Example of matrix multiplication in OpenMP.
 *
 * (C) 2015 Mikhail Kurnosov <mkurnosov@gmail.com>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <sys/time.h>

/* A[n, m] * B[m, q] = C[n, q] */
enum {
    N = 2000,   
    M = 2000,
    Q = 2000
};
    
double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

/* dgemm_def: Naive matrix multiplication C[n, q] = A[n, m] * B[m, q] */
void dgemm_def(double *a, double *b, double *c, int n, int m, int q)
{
    /* FP ops: 2 * n * q * m */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < q; j++) {
            double s = 0.0;
            for (int k = 0; k < m; k++)
                s += a[i * m + k] * b[k * q + j];
            c[i * q + j] = s;
		}
	}
}

/* dgemm_opt: Matrix multiplication C[n, q] = A[n, m] * B[m, q] */
void dgemm_opt(double *a, double *b, double *c, int n, int m, int q)
{
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

/* dgemm_opt: Matrix multiplication C[n, q] = A[n, m] * B[m, q] */
void dgemm_opt_omp(double *a, double *b, double *c, int n, int m, int q)
{
    #pragma omp parallel
    {
    int k = 0;
    #pragma omp for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < q; j++)
            c[k++] = 0;

    #pragma omp for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            for (int j = 0; j < q; j++)
                c[i * q + j] += a[i * m + k] * b[k * q + j];
		}
	}
	}
}

int main(int argc, char **argv)
{
    double gflop = 2.0 * N * Q * M * 1E-9;
    double *a, *b, *c1, *c2;
    printf("DGEMM N = %d, M = %d, Q = %d\n", N, M, Q);
    
    // Launch naive verstion    
    a = malloc(sizeof(*a) * N * M);
    b = malloc(sizeof(*b) * M * Q);
    c1 = malloc(sizeof(*c1) * N * Q);
    if (a == NULL || b == NULL || c1 == NULL) {
        fprintf(stderr, "No enough memory\n");
        exit(1);
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

    double tdef = wtime();
    dgemm_def(a, b, c1, N, M, Q);
    tdef = wtime() - tdef;
    printf("Execution time (naive): %.6f\n", tdef);
    printf("Performance (naive): %.2f GFLOPS\n", gflop / tdef);
    free(b);
    free(a);        

    // Launch opt. version
    a = malloc(sizeof(*a) * N * M);
    b = malloc(sizeof(*b) * M * Q);
    c2 = malloc(sizeof(*c2) * N * Q);
    if (a == NULL || b == NULL || c2 == NULL) {
        fprintf(stderr, "No enough memory\n");
        exit(1);
    }

    srand(0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++)
            a[i * M + j] = rand() % 100;      // 1.0;
	}
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < Q; j++)
            b[i * Q + j] = rand() % 100;         // 2.0;
	}

    double topt = wtime();
    dgemm_opt_omp(a, b, c2, N, M, Q);
    topt = wtime() - topt;
    printf("Execution time (opt): %.6f\n", topt);
    printf("Performance (opt): %.2f GFLOPS\n", gflop / topt);
    free(b);
    free(a);        
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < Q; j++)
            if (fabs(c1[i * Q + j] - c2[i * Q + j]) > 1E-6) {
                fprintf(stderr, "ERROR: Invalid result\n");
                exit(1);
            }
	}
    free(c2);
    free(c1);
    
    printf("Speedup: %.2f\n", tdef / topt);
        
    return 0;
}
