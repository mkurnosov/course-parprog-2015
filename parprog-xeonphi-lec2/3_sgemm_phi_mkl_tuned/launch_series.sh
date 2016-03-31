#!/bin/sh

# Xeon Phi OpenMP
export MIC_ENV_PREFIX=MIC
export MIC_USE_2MB_BUFFERS=10M
export MIC_KMP_AFFINITY=explicit,granularity=fine,proclist=[1-224:1]

for i in `seq 448 56 800`; do
    export MIC_OMP_NUM_THREADS=$i
    ./sgemm
done
