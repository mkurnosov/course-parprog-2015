#!/bin/sh

export OMP_NUM_THREADS=2
valgrind --tool=helgrind ./integrate


