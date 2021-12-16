#!/bin/sh

date

cd ~
home_folder=$(pwd)
agent=$1
world=$2

source $home_folder/python_envs/venv/bin/activate

python $home_folder/Documents/PhD/legible_mdp/src/irl_simulation.py --agent $agent --world world --leg_func leg_optimal --fail_prob 0.15 --mode sample --reps 250 --no_verbose --batch_size 20

deactivate

date