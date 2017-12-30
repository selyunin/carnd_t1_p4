'''
Created on Dec 27, 2017
@author: selyunin
'''

from moviepy.editor import VideoFileClip
from IPython.display import HTML
from collections import deque
from ImageHandler import ImageHandler

class VideoHandler():
    
    def __init__(self, *args, **kwargs):
        self.clip_name = args[0].input_video
        self.subclip_length = args[0].subclip_length
        self.out_clip = None
        self.out_clip_name = args[0].output_video
#         self.video_clip = VideoFileClip(self.clip_name).subclip(0, self.subclip_length)
#         self.video_clip = VideoFileClip(self.clip_name).subclip(20, 25)
#         self.video_clip = VideoFileClip(self.clip_name).subclip(38, 40)
        self.video_clip = VideoFileClip(self.clip_name)
        self.frame_counter = 0
        self.image_handler = ImageHandler()
    
    def process_video(self):
        self.out_clip = self.video_clip.fl_image(self.process_image) #NOTE: this function expects color images!!
        self.out_clip.write_videofile(self.out_clip_name, audio=False)
    
    def process_image(self, img):
        self.frame_counter += 1
        out_img = self.image_handler.process_image(img)
#         print("Processing frame: {} of shape {}".format(self.frame_counter, img.shape))
        return out_img