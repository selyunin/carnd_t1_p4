#!/usr/bin/env python

'''
Created on Dec 27, 2017
@author: selyunin
'''

import os
import sys
import cv2
import glob
import numpy as np
import datetime
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from collections import OrderedDict

from VideoHandler import VideoHandler

def process_video(args):
    print("Starting video processing")
    for key, value in (vars(args)).items():
        print("{:15s} -> {}".format(key, value))
    
    video_handler = VideoHandler(args)
    print("video_handler: {}".format(video_handler.clip_name))
    video_handler.process_video()
    pass

def process_video_v(args):
    print("Starting video processing")
    args.input_video = 'project_video.mp4'
    args.output_video = 'project_video_out.mp4'
    args.subclip_length = 9
    for key, value in (vars(args)).items():
        print("{:15s} -> {}".format(key, value))
    
    video_handler = VideoHandler(args)
    print("video_handler: {}".format(video_handler.clip_name))
    video_handler.process_video()

def main():
    description = "Advanced Line Lines Python project" 
    time_now = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input_video", type=str, default='project_video.mp4')
    parser.add_argument("-o", "--output_video", type=str, default='project_video_out_{}.mp4'.format(time_now))
    parser.add_argument("-s", "--subclip_length", type=int, default=7)
    args = parser.parse_args(sys.argv[1:])
    
    print("args == ")
    print(args)
    process_video(args)
    
if __name__ == '__main__':
    main()