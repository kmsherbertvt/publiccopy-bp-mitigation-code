echo "Setting up environment"
module load Julia/1.7.2-linux-x86_64

export NTHREAD=20
export JULIAENV=/home/gbarron/bp-mitigation-code/
# set these to 1 if you are using julia threads heavily to avoid oversubscription
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

#echo "Usage: sbatch job.sh script.jl"
#export INFILE=$1
#export OUTFILE="${INFILE}.out"
#echo $INFILE
#echo $OUTFILE

rm output_*.txt

rm ./figs/*.pdf
rm ./data/*.csv

if $INTERACTIVE_JOB ; then
    echo "Starting interactive job..."
    julia --project=$JULIAENV
else
    echo "Starting non-interactive job..."
    julia --project=$JULIAENV --procs=auto bp_sampling.jl $SCRIPT_DEBUG >& output_sims.txt
    julia --project=$JULIAENV plotting.jl $SCRIPT_DEBUG >& output_plots.txt
fi

exit