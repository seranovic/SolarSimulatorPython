#!/bin/bash
#SBATCH --account DeiC-RUC-L2-202601
#SBATCH -c 32
#SBATCH --mem 8g
#SBATCH -t 7200

# -c amount of logical cores (threads)
# --mem amount of memory
# -t minimum time in minutes (should be 480 per 1e4x5e4 run)

offset=1
runs=10
outersteps=10000
innersteps=50000

source env/bin/activate

for i in $(seq $((1 + $offset)) $(($runs + $offset))); do
    echo "ID: $i - 2 stars"
    python init_cond_2_stars.py $i
    python run_script.py $i $outersteps $innersteps
done

for i in $(seq $((1 + $offset + $runs)) $(($runs + $offset + $runs))); do
    echo "ID: $i - 1 star"
    python generator.py $i
    python run_script.py $i $outersteps $innersteps
done

deactivate
