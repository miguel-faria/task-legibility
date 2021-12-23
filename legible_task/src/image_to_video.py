import sys

import cv2
import numpy as np
import os
import argparse
import re
import pathlib

from pathlib import Path


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def main():

    parser = argparse.ArgumentParser(description='Convert WeBots simulation snippets to video')
    parser.add_argument('--mode', type=str, required=True, help='Type of action selection used')
    parser.add_argument('--target', type=str, required=True, help='Agent\'s objective')
    parser.add_argument('--world', type=int, required=True, help='Environment recorded')

    args = parser.parse_args()

    print('[IMAGE TO VIDEO] Loading photo stills of the movement')
    
    video_dir = 'w' + str(args.world) + '_' + args.mode + '_' + args.target
    if str(sys.platform).find('linux') != -1:
        image_dir = Path('/mnt/c/Users/migue/Documents/WeBots/task_legibility/controllers/camera/cam_robot') / video_dir
    else:
        image_dir = Path('C:/Users/migue/Documents/WeBots/task_legibility/controllers/camera/cam_robot') / video_dir
    script_parent_dir = Path(__file__).parent.absolute().parent.absolute()
    videos_dir = script_parent_dir / 'data' / 'webots_videos'
    img_array = []
    img_files = os.listdir(image_dir)
    img_files.sort(key=natural_keys)

    size = ()
    print('[IMAGE TO VIDEO] Create list sequence from the images')
    for filename in img_files:
        img_path = image_dir / filename
        if not img_path.is_dir():
            img = cv2.imread(str(img_path))
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
          
    print('[IMAGE TO VIDEO] Creating video from the images')
    if len(size) > 0:
        out = cv2.VideoWriter(str(videos_dir / ('w' + str(args.world) + '_' + args.mode + '_movement_' + args.target + '.mp4')),
                              cv2.VideoWriter_fourcc('V','P','8','0'), 11.5, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    else:
        print('[IMAGE TO VIDEO] Error creating video, no frame size defined!!')


if __name__ == '__main__':
    main()