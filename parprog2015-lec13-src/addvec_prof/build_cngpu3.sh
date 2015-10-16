#$!/bin/sh

nvcc -O2 -o vadd ./vadd.cu -arch=compute_11 -code=sm_11

