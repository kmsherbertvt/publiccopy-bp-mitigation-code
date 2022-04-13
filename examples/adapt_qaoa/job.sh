#!/usr/bin/env bash

#SBATCH -p normal_q
#SBATCH -A qc_group
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10gb
#SBATCH -t 01:00:00

module load Julia/1.7.2-linux-x86_64

julia --threads=auto --project=@. adapt_qaoa.jl

wait
exit 0
