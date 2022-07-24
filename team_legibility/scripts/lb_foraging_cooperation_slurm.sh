#!/bin/bash

# Script to execute evaluation of legible cooperation in the LB-Foraging environment in slurm environments

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --time=336:00:00
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

source $CLUSTER_HOME/python_envs/lb_env/bin/activate

python $CLUSTER_HOME/team_legibility/src/lbforaging_legible_cooperation.py --runs $nruns --mode ${team_comps[@]} --paralell --nagents $nagents --nfood $nfood --field_length $field_length

deactivate

date