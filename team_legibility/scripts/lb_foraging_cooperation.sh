#!/bin/bash

# Script to execute evaluation of legible cooperation in the LB-Foraging environment

date;hostname;pwd

args=($@)
nruns=$1
nagents=$2
field_length=$3
nfood=$4

team_comps=()
for ((i=4; i<=$#; i++))
do
  team_comps+=(${args[$i]})
done

source $HOME/python_envs/lb_env/bin/activate

python $HOME/Documents/PhD/team_legibility/src/lbforaging_legible_cooperation.py --runs $nruns --mode ${team_comps[@]} --paralell --nagents $nagents --nfood $nfood --field_length $field_length

deactivate

date