#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define NELEMS(x) (sizeof(x) / sizeof((x)[0]))

enum {
    NREPS = 5,
};
    
double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

void testbw(int size, int alignment)
{
    uint8_t *buf = _mm_malloc(size, alignment);
    if (!buf) {
        fprintf(stderr, "Can't allocate memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Init buffer (allocate pages)
    memset(buf, 1, size);
    double t, talloc;
    double tsend = 0.0;
    double trecv = 0.0;
    
    // Allocate buffer on Phi
    talloc = wtime();
    #pragma offload target(mic) in(buf:length(size) free_if(0))
    { 
        buf[2] = size;
    }
    talloc = wtime() - talloc;
    
    // Measures
    for (int i = 0; i < NREPS; i++) {
        // Copy to Phi
        t = wtime();
        #pragma offload target(mic) in(buf:length(size) alloc_if(0) free_if(0))
        { 
            buf[4] = size;
        }
        tsend += wtime() - t;
        
        // Copy from Phi
        t = wtime();
        #pragma offload target(mic) out(buf:length(size) alloc_if(0) free_if(0))
        { 
            buf[9] = size;
        }
        trecv += wtime() - t;        
    }    

    // Free on Phi
    #pragma offload target(mic) in(buf:length(size) alloc_if(0) free_if(1))
    { 
        buf[1] = 3;
    }

    tsend /= NREPS;
    trecv /= NREPS;
    
    printf("%-10d  %-8d  %-.6f  %-.6f  %-.6f\n", size, alignment, talloc, tsend, trecv);    
    _mm_free(buf);
}

int main(int argc, char **argv)
{
    printf("Xeon Phi bwtest: nreps = %d\n", NREPS);
    printf("size        alignment  talloc  tsend  trecv\n");
        
    //int s[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int s[] = {64};
    int a[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    
    for (int j = 0; j < NELEMS(a); j++) {
        for (int i = 0; i < NELEMS(s); i++) {
            testbw(s[i] * 1024 * 1024, a[j]);
        }
    }
    return 0;
}
