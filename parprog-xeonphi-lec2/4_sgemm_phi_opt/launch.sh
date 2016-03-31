#!/bin/sh

# Xeon Phi OpenMP
export MIC_ENV_PREFIX=MIC
export MIC_OMP_NUM_THREADS=672

export MIC_USE_2MB_BUFFERS=10M

export MIC_KMP_AFFINITY=explicit,granularity=fine,proclist=[1-224:1]
./sgemm

#export MIC_KMP_AFFINITY=granularity=fine,balanced
#./sgemm

#export MIC_KMP_AFFINITY=granularity=thread,balanced
#./sgemm

#export MIC_KMP_AFFINITY=granularity=fine,scatter
#./sgemm

