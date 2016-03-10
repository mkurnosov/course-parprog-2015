#!/bin/sh

clang -Wall -O2 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize ./vec.c -ovec
