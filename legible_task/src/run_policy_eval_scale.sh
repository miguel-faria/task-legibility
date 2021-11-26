#!/bin/bash
cd ~
date;hostname;pwd

home_folder=$(pwd)

source $home_folder/python_envs/venv/bin/activate

python $home_folder/Documents/PhD/legible_mdp/src/legible_evaluation.py --framework policy --evaluation scale --metric all --reps 250 --fail_prob 0.15 --beta 0.5 --gamma 0.9

deactivate

date
