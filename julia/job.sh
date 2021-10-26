#!/usr/bin/env bash

#SBATCH -J mcp-rand-ham
#SBATCH -p normal_q
#SBATCH -A jvandyke_alloc
#SBATCH -n 1
#SBATCH --cpus-per-task=120
#SBATCH --mem=240gb
#SBATCH -t 4:00:00

module load Julia/1.5.1-linux-x86_64

julia -p auto bp_adapt.jl

wait
exit 0