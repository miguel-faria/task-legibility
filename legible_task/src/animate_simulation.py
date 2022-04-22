import sys

import cv2
import os
import argparse
import re

from pathlib import Path


def atoi(text):
	return int(text) if text.isdigit() else text


def natural_keys(text):
	return [atoi(c) for c in re.split('(\d+)', text)]


def main():
	parser = argparse.ArgumentParser(description='Convert animation stills into full animations')
	parser.add_argument('--mode', type=str, required=True, help='Type of action selection used')
	
	args = parser.parse_args()
	mode = args.mode
	
	print('[ANIMATE SIMULATION] Loading photo stills of the movement')
	
	script_parent_dir = Path(__file__).parent.absolute().parent.absolute()
	image_dir = script_parent_dir / 'data' / 'simulation_stills'
	videos_dir = script_parent_dir / 'data' / 'simulation_videos'
	img_array = []
	img_files = os.listdir(image_dir)
	img_files.sort(key=natural_keys)
	
	size = ()
	print('[ANIMATE SIMULATION] Create list sequence from the images')
	for filename in img_files:
		if filename.find(mode) != -1:
			img_path = image_dir / filename
			if not img_path.is_dir():
				img = cv2.imread(str(img_path))
				height, width, layers = img.shape
				size = (width, height)
				img_array.append(img)
	
	print('[ANIMATE SIMULATION] Creating video from the images')
	if len(size) > 0:
		out = cv2.VideoWriter(str(videos_dir / ('simulation_animation_' + args.mode + '_movement.mp4')),
							  cv2.VideoWriter_fourcc('V', 'P', '8', '0'), 10.0, size)
		
		for i in range(len(img_array)):
			out.write(img_array[i])
		out.release()
	
	else:
		print('[IMAGE TO VIDEO] Error creating video, no frame size defined!!')


if __name__ == '__main__':
	main()