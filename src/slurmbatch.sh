#! /bin/bash
#
#SBATCH --ntasks=8
for i in {1..50}
do
   srun -N1 -n1 -c1 --exclusive ./python run_scr_estimation.py --config_file ../data/simulation/config/lowfrag_N100_a2225.csv --ac_realization_no 0 --caphist_realization_no $i &
done
wait
