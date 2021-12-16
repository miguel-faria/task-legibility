#!/usr/bin/env python

import yaml
import argparse

from pathlib import Path


def main():
	parser = argparse.ArgumentParser(description='Consistency tester for the project\'s YAML config files')
	parser.add_argument('--filename', dest='filename', type=str, required=True,
						help='YAML file name')
	
	args = parser.parse_args()
	filename = args.filename
	
	script_parent_dir = Path(__file__).parent.absolute().parent.absolute()
	config_dir = script_parent_dir / 'data' / 'configs'
	
	file_path = config_dir / (filename + '.yaml')
	with open(file_path, 'r') as file:
		
		config_params = yaml.full_load(file)
		for param in config_params:
			print(param)
			print(config_params[param])
			print('\n')
	

if __name__ == '__main__':
	main()