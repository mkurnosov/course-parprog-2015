#!/bin/sh

# Xeon Phi OpenMP
export MIC_ENV_PREFIX=MIC

for i in `seq 56 56 448`; do
    export MIC_OMP_NUM_THREADS=$i
    ./sgemm
done
