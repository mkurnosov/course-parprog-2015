#!/bin/sh

for i in `seq 2 4 256`; do
    export OMP_NUM_THREADS=$i
    ./sgemm OMP-ONLY
done
