#!/bin/bash

# Script to generate the legible decision models in slurm environments

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --time=336:00:00
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

source $CLUSTER_HOME/python_envs/lb_env/bin/activate

python $CLUSTER_HOME/team_legibility/src/lbforaging_generate_legible_behavior.py --rows $field_size --cols $field_size --max_food 2 --food_level 2 --agents $agents --agent_levels ${agent_levels[@]} --agent_mode leader &
python $CLUSTER_HOME/team_legibility/src/lbforaging_generate_legible_behavior.py --rows $field_size --cols $field_size --max_food 2 --food_level 2 --agents $agents --agent_levels ${agent_levels[@]} --agent_mode follower &

deactivate

date
