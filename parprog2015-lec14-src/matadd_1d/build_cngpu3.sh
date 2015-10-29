#$!/bin/sh

nvcc -O2 -o matadd ./matadd.cu -arch=compute_11 -code=sm_11


