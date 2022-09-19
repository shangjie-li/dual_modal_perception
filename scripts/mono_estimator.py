# -*- coding: UTF-8 -*- 

import numpy as np
import math
import cv2
from math import sin, cos

class MonoEstimator():
    def __init__(self, file_path, print_info=True):
        fs = cv2.FileStorage(file_path, cv2.FileStorage_READ)
        
        mat = fs.getNode('ProjectionMat').mat()
        self.fx = int(mat[0, 0])
        self.fy = int(mat[1, 1])
        self.u0 = int(mat[0, 2])
        self.v0 = int(mat[1, 2])
        
        self.height = fs.getNode('Height').real() # meter
        self.depression = fs.getNode('DepressionAngle').real() # degree
        
        if print_info:
            print('Calibration of camera:')
            print('  Parameters: fx(%d) fy(%d) u0(%d) v0(%d)' % (self.fx, self.fy, self.u0, self.v0))
            print('  Height: %.2fm' % self.height)
            print('  DepressionAngle: %.2fdegree' % self.depression)
            print()
    
    def get_location(self, u, v):
        theta_0 = self.depression * np.pi / 180
        theta_lat = np.arctan2(u - self.u0, self.fx)
        theta_ver = theta_0 + np.arctan2(v - self.v0, self.fy)
        if theta_ver < 1e-5:
            return (float('inf'), float('inf'))
        else:
            range_lon = self.height / np.tan(theta_ver)
            range_lat = range_lon * np.tan(theta_lat)
            return (range_lat, range_lon)
    
    def estimate(self, box):
        box = box.reshape(-1)
        u, v = (box[0] + box[2]) / 2, box[3]
        return self.get_location(u, v)
    
