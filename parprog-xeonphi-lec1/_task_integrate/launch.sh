#!/bin/sh

# Host OpenMP
export OMP_NUM_THREADS=12

# Xeon Phi OpenMP
export MIC_ENV_PREFIX=MIC
export MIC_OMP_NUM_THREADS=56
#export MIC_KMP_AFFINITY=verbose,granularity=fine,balanced
#export MIC_KMP_AFFINITY=verbose,granularity=fine,compact
#export MIC_KMP_AFFINITY=verbose,granularity=fine,scatter

./integrate
