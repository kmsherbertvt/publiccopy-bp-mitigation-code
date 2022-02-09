#!/bin/bash

out_file='./job.sh.out'
rm $out_file

start_time=$SECONDS
echo "Starting job at $(date)" >> $out_file


SEEDS=50
NMAX=14
echo "Running simulations" >> $out_file
#julia --threads auto --project=.. bp_adapt.jl $NMAX $SEEDS >> $out_file

echo "Generating plots" >> $out_file
python data_analysis.py

elapsed=$(( SECONDS - start_time ))
echo "Done! with NMAX=$NMAX, SEEDS=$SEEDS" >> $out_file

eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')" >> $out_file

wait
exit 0
