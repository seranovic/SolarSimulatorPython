#!/bin/bash
#SBATCH --account DeiC-RUC-L2-202601
#SBATCH -c 32
#SBATCH --mem 16g

source env/bin/activate

for i in $(seq 1 3); do
    python init_cond_2_stars.py $i
    python run_script.py $i 10000 50000
done

deactivate
