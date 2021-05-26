#!/usr/bin/env bash

#SBATCH -J mcp-rand-ham
#SBATCH -p normal_q
#SBATCH -A jvandyke_alloc
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
##SBATCH --array=0-9
#SBATCH -t 143:00:00

module load Julia/1.5.1-linux-x86_64

julia bp_adapt.jl

wait
exit 0
