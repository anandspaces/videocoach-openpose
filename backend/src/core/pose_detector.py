"""
Pose Detection Module
Handles pose detection using OpenPose/Caffe model
"""

import cv2
import numpy as np


class PoseDetector:
    def __init__(self, model_file, config_file, use_cuda=False):
        self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.input_size = (368, 368)
        
        # COCO keypoint pairs
        self.pose_pairs = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [8,9], 
                          [9,10], [1,11], [11,12], [12,13], [0,14], [0,15], [14,16], [15,17]]
        
        self.points_names = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 
                           'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 
                           'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar']
    
    def detect(self, frame):
        height, width = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, self.input_size, 
                                     (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        
        try:
            output = self.net.forward()
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return [], []
        
        H, W = output.shape[2:]
        
        points = [None] * 18
        points_prob = []
        
        for idx in range(18):
            prob_map = output[0, idx, :, :]
            
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            
            x = (width * point[0]) / W
            y = (height * point[1]) / H
            
            if prob > 0.05:
                points[idx] = (x, y, prob)
            else:
                points[idx] = None
            
            points_prob.append(prob)
        
        return points, points_prob
