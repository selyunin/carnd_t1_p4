'''
Created on Dec 27, 2017
@author: selyunin
'''
# Define a class to receive the characteristics of each line detection
from collections import deque
import numpy as np

class Line():
    def __init__(self):
        self.N_WINDOW = 13
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        self.poly_fit  = deque([], self.N_WINDOW)
        self.poly_y    = deque([], self.N_WINDOW)
        self.poly_x    = deque([], self.N_WINDOW)
        self.curvature = deque([], self.N_WINDOW)
        self.offset    = deque([], self.N_WINDOW)
        self.x_min     = deque([], self.N_WINDOW)
        self.x_max     = deque([], self.N_WINDOW)
    
    def set_current_poly_fit(self, poly_fit, poly_y, poly_x):
        y_max = max(poly_y)
        y_min = min(poly_y)
        x_max = poly_fit[0]* y_max**2 + \
                poly_fit[1]* y_max + \
                poly_fit[2]
                
        x_min = poly_fit[0]* y_min**2 + \
                poly_fit[1]* y_min + \
                poly_fit[2]
        if len(self.x_min) == 0 :
            self.poly_fit.append(poly_fit)
            self.poly_y.append(poly_y)
            self.poly_x.append(poly_x)
            self.x_min.append(x_min)
            self.x_max.append(x_max)
        elif abs((self.x_min[-1] - x_min)/self.x_min[-1]) <= 0.15 and \
             abs((self.x_max[-1] - x_max)/self.x_max[-1]) <= 0.15 :
            self.poly_fit.append(poly_fit)
            self.poly_y.append(poly_y)
            self.poly_x.append(poly_x)
            self.x_min.append(x_min)
            self.x_max.append(x_max)


    def set_current_curvature(self, rad):
        self.curvature.append(rad)
        
    def set_current_offset(self, offset):
        self.offset.append(offset)
        
    def get_curvature(self):
        return sum(self.curvature)/self.N_WINDOW
    
    def get_offset(self):
        return sum(self.offset)/self.N_WINDOW
    
    def get_poly(self):
        return self.poly_y[-1], self.poly_x[-1]
#         return np.average(self.poly_y), np.average(self.poly_x)