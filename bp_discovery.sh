#!/usr/bin/env bash

#SBATCH -J dask-nanny
#SBATCH -p normal_q
#SBATCH -A jvandyke_alloc
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20gb
#SBATCH -t 24:00:00

touch 1.out
touch 1.log
rm *.out
rm *.log

source activate barren
module load dask/2.18.1-foss-2020a-Python-3.8.2

python bp_discovery.py

wait
exit 0
