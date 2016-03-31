#!/bin/sh

# Xeon Phi OpenMP
export MIC_ENV_PREFIX=MIC
export MIC_OMP_NUM_THREADS=224
#export MIC_KMP_AFFINITY=verbose,granularity=fine,balanced

./sgemm

