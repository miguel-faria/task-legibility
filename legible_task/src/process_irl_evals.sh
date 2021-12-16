#!/bin/bash

date

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
script_dir_parent="$( cd -- "$script_dir" >/dev/null 2>&1 ; cd ..; pwd -P )"
irl_results=$(ls "$script_dir_parent/data/results/irl_performance")

for result in $irl_results
do
	if ! [ -d $result ] && { [[ $result = *"_legible.csv" ]] || [[ $result = *"_optimal.csv" ]]; };
	then
		result_name=${result%%.*}
		python "$script_dir/process_irl_eval.py" --filename $result_name --folder-path "results/irl_performance"
	fi
done

date