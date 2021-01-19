#!/usr/bin/env bash

#SBATCH -J dask-worker
#SBATCH -p dev_q
#SBATCH -A personal
#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH -t 00:05:00

source activate barren

python bp_discovery.py