#!/bin/bash

# Job Admin Stuff
#SBATCH --mail-user=mute-arcjobs@saem.xyz
#SBATCH --mail-type=ALL
#SBATCH -A qc_group
#SBATCH -p normal_q

# Resources
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH -t 72:00:00

# Allow time to load modules
# This takes time sometimes for some reason
sleep 10
hostname
#module reset
module load Julia/1.7.2-linux-x86_64

export NTHREAD=20
export JULIAENV=/home/gbarron/bp-mitigation-code/
# set these to 1 if you are using julia threads heavily to avoid oversubscription
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Usage: sbatch job.sh script.jl"
export INFILE=$1
export OUTFILE="${INFILE}.out"
echo $INFILE
echo $OUTFILE

julia --project=$JULIAENV -t $NTHREAD $INFILE >& $OUTFILE
#julia --project=$JULIAENV --procs=auto $INFILE >& $OUTFILE

exit
