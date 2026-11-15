#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cat isler.txt | xargs -P 4 -L 1 python kos.py >> kos.log 2>&1
echo BITTI >> kos.log
