#$!/bin/sh

nvcc -O2 -arch sm_30 -o sgemm ./sgemm.cu


