#$!/bin/sh

#nvcc -O2 -arch sm_30 --use_fast_math -o tabfun ./tabfun.cu -lm
nvcc -O2 -arch sm_30 -o tabfun ./tabfun.cu -lm



