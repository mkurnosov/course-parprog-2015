#!/bin/sh

gcc -Wall -std=c99 -O2 -march=native -ftree-vectorize -fopt-info-vec -fopt-info-vec-missed ./vec.c -ovec
