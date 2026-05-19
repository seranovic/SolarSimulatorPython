#!/bin/bash
#SBATCH --account DeiC-RUC-L2-202601
#SBATCH -c 32
#SBATCH --mem 8g
#SBATCH -t 10080

# -c amount of logical cores (threads)
# --mem amount of memory
# -t minimum time in minutes (should be 480 per 1e4x5e4 run)

offset=1
runs=20
outersteps=10000
innersteps=50000

source env/bin/activate

for i in $(seq $((1 + $offset)) $(($runs + $offset))); do
    echo "ID: $i"
    python init_cond_2_stars.py $i
    python run_script.py $i $outersteps $innersteps
done

deactivate
