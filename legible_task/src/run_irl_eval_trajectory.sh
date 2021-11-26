#!/bin/sh

date;

cd ~
home_folder=$(pwd)
agent=$1

source $home_folder/python_envs/venv/bin/activate

python $home_folder/Documents/PhD/legible_mdp/src/irl_simulation.py --agent $agent --world 5 --leg_func leg_optimal --fail_prob 0.15 --mode trajectory --reps 250 --no_verbose --traj_len 20

deactivate

date