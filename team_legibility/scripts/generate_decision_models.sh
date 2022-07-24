#!/bin/bash

# Script to generate the legible decision models

date;hostname;pwd

args=($@)
field_size=$1
agents=$2

agent_levels=()
for ((i=0; i<=$agents; i++))
do
  arg_idx=$(($i + $agents))
  agent_levels+=(${args[$arg_idx]})
done

source $HOME/python_envs/lb_env/bin/activate

python $HOME/Documents/PhD/team_legibility/src/lbforaging_generate_legible_behavior.py --rows $field_size --cols $field_size --max_food 2 --food_level 2 --agents $agents --agent_levels ${agent_levels[@]} --agent_mode leader
python $HOME/Documents/PhD/team_legibility/src/lbforaging_generate_legible_behavior.py --rows $field_size --cols $field_size --max_food 2 --food_level 2 --agents $agents --agent_levels ${agent_levels[@]} --agent_mode follower

deactivate

date
