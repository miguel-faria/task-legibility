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
	parser.add_argument('--filename', dest='filename', type=str, required=True,
						help='CSV file name')
	parser.add_argument('--folder-path', dest='folder', type=str,
						help='Path from \'data\' folder to subfolder with csv file')
		
	args = parser.parse_args()
	filename = args.filename
	file_folder = args.folder
	
	print('Processing file: %s' % filename)
	script_parent_dir = Path(__file__).parent.absolute().parent.absolute()
	data_dir = script_parent_dir / 'data'
	
	file = data_dir
	out_file = data_dir
	if file_folder:
		subfolders = file_folder.split('/')
		for folder in subfolders:
			file /= folder
			out_file /= folder
		file /= (filename + '.csv')
		out_file /= (filename + '_averages.csv')
	else:
		file /= (filename + '.csv')
		out_file /= (filename + '_averages.csv')
		
	csv_contents = {}
	with open(file, 'r') as data:
		for line in csv.DictReader(data):
			keys = line.keys()
			for key in keys:
				if key.find('trajectory') == -1 and key.find('iteration') == -1:
					if key not in csv_contents:
						csv_contents[key] = [np.asarray(re.sub('[ ]+', ' ', line[key]).replace('[', '').replace(']', '').rstrip().split(' '), dtype=np.float64)]
					else:
						csv_contents[key] += [np.asarray(re.sub('[ ]+', ' ', line[key]).replace('[', '').replace(']', '').rstrip().split(' '), dtype=np.float64)]
	
	with open(out_file, 'w') as out:
		fields = ','.join(['key', 'average', 'std_err'])
		out_contents = []
		for key in csv_contents.keys():
			average = np.average(csv_contents[key], axis=0)
			std_err = np.std(csv_contents[key], axis=0) / math.sqrt(len(csv_contents[key]))
			print(key)
			print(average)
			print(std_err)
			print('\n')
			out_contents += [[key, average, std_err]]
		out_contents = np.array(out_contents, dtype=object)
		np.savetxt(fname=out, X=out_contents, delimiter=',', header=fields, fmt='%s', comments='')
	

if __name__ == '__main__':
	main()
