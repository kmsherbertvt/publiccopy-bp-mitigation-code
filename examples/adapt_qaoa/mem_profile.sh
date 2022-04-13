julia --threads=auto --project=@. adapt_qaoa.jl &
pid=$!
touch mem.txt
rm mem.txt
while true;
do
    grep ^VmPeak /proc/$pid/status >> mem.txt || { echo 'Done!' ; exit 1; }
    sleep 3
done