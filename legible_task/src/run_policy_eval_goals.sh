#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --time=336:00:00
date;hostname;pwd

module load python/3
source /mnt/cirrus/users/9/2/ist173092/python_envs/venv/bin/activate

python /mnt/cirrus/users/9/2/ist173092/legible_task/src/legible_evaluation.py --framework policy --evaluation goals --metric all --reps 250 --fail_prob 0.15 --beta 0.5 --gamma 0.9
cp /mnt/cirrus/users/9/2/ist173092/legible_task/logs/evaluation_log_policy_goals_all* /afs/ist.utl.pt/users/9/2/ist173092/Desktop/legible_task/logs/
cp /mnt/cirrus/users/9/2/ist173092/legible_task/data/results/evaluation_results_policy_goals_all.csv /afs/ist.utl.pt/users/9/2/ist173092/Desktop/legible_task/data/results/

deactivate

date
