#!/usr/bin/env python

import sys
import os
import re

import numpy as np
import argparse
import csv
import math

np.set_printoptions(precision=5, linewidth=1000)

from typing import Dict, List, Tuple
from pathlib import Path


def main():
	parser = argparse.ArgumentParser(description='CSV processing script')
	parser.add_argument('--folder-path', dest='folder', type=str, required=True,
						help='Path from \'data\' folder to subfolder with csv files')
	
	args = parser.parse_args()
	file_folder = args.folder
	
	script_parent_dir = Path(__file__).parent.absolute().parent.absolute()
	data_dir = script_parent_dir / 'data'
	
	files_dir = data_dir
	subfolders = file_folder.split('/')
	for folder in subfolders:
		files_dir /= folder
	
	csv_contents = []
	for path in files_dir.iterdir():
		if path.is_file() and str(path).find('answers') != -1:
			file = path
			with open(file, 'r') as data:
				csv_reader = csv.reader(data)
				next(csv_reader)
				for row in csv_reader:
					csv_contents += [row]
	
	out_file = files_dir / 'answers_join.csv'
	with open(out_file, 'w+', newline='') as out:
		write = csv.writer(out)
		write.writerows(csv_contents)


if __name__ == '__main__':
	main()
