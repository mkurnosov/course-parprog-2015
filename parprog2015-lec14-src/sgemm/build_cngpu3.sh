#$!/bin/sh

nvcc -O2 -o sgemm ./sgemm.cu -arch=compute_11 -code=sm_11


