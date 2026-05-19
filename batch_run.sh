#!/bin/bash
#SBATCH --account DeiC-RUC-L2-202601
#SBATCH -c 32
#SBATCH --mem 8g
#SBATCH -t 7200

# -c amount of logical cores (threads)
# --mem amount of memory
# -t minimum time in minutes (should be 480 per 1e4x5e4 run)

outersteps=$1 # 10000
innersteps=$2 # 50000
runs=$3 # 10
offset=$4 # the total amount of datasets from completed and ongoing runs

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
