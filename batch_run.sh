#!/bin/bash
#SBATCH --account DeiC-RUC-L2-202601
#SBATCH -c 32
#SBATCH --mem 16g
#SBATCH -t 960

# -c amount of logical cores (threads)
# --mem amount of memory
# -t minimum time in minutes (should be 480 per 1e4x5e4 run)

runs=2
outersteps=10000
innersteps=50000

source env/bin/activate

for i in $(seq 1 $runs); do
    python init_cond_2_stars.py $i
    python run_script.py $i $outersteps $innersteps
done

deactivate
