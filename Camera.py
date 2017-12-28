'''
@date: Dec 21, 2017
@author: selyunin
@license: MIT
@version: 0.1
'''

import cv2
import pickle

class Camera:
    
    def __init__(self, pickle_file = 'camera_pickle.p'):
        self.pickle = pickle_file
        with(open(self.pickle, 'rb')) as p:
            camera_pickle = pickle.load(p)
            self.mtx = camera_pickle["mtx"]
            self.dist = camera_pickle["dist"]
    
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        