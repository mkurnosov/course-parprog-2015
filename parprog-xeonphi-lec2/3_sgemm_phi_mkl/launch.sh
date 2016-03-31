#!/bin/sh

# Xeon Phi OpenMP
export MIC_ENV_PREFIX=MIC
export MIC_OMP_NUM_THREADS=224

./sgemm

#export MIC_KMP_AFFINITY=granularity=fine,balanced
#./sgemm

#export MIC_KMP_AFFINITY=granularity=thread,balanced
#./sgemm

#export MIC_KMP_AFFINITY=granularity=fine,scatter
#./sgemm

