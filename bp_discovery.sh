#!/usr/bin/env bash

#SBATCH -J dask-nanny
#SBATCH -p normal_q
#SBATCH -A personal
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00

source activate barren
module load dask/2.18.1-foss-2020a-Python-3.8.2

python bp_discovery.py

wait
exit 0