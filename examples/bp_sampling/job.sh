#!/bin/bash

# Job Admin Stuff
#SBATCH --mail-user=gbarron+ping@vt.edu
#SBATCH --mail-type=END,FAILED
#SBATCH -A qc_group
#SBATCH -p normal_q

# Resources
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH -t 72:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

# Allow time to load modules
# This takes time sometimes for some reason
sleep 10
hostname

export SCRIPT_DEBUG="full_run"
export INTERACTIVE_JOB=false

./run.sh