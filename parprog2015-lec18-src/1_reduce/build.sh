#$!/bin/sh

nvcc -O2 -arch sm_30 -o reduction ./reduction.cu -lm




