#!/usr/bin/env bash

#SBATCH --mail-user=mute-arcjobs@saem.xyz
#SBATCH --mail-type=ALL
#SBATCH -p normal_q
#SBATCH -A qc_group
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10gb
#SBATCH -t 01:00:00

module load Julia/1.7.2-linux-x86_64

julia --project=@. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --threads=auto --project=@. adapt_qaoa.jl

wait
exit 0
