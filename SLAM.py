import cv2
import numpy as np
import os.path as osp
from Map import Map
from Geometry import *
from Feature import *
from Database import Dataset
from PointCloud import PointCloud

class SLAM(object):
    def __init__(self,root,fps = 25):
        self.frameCnt = 0
        self.Dataset = Dataset(root)
        self.Map = Map()
        self.A = np.genfromtxt(osp.join(root, "cam.txt"), delimiter=',')
        self.fps = fps
        self.transform = np.eye(4,4)
        self.KF = Kalman(1.0/self.fps)
        self.prevImg =None
        self.prevDepth = None
        self.prevFeat = None
        self.PC = PointCloud()
        self.feat = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_KAZE)

    def addFrame(self,img,depth):
        # Detect features
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp = self.feat.detect(img_gray)
        kp,desc = self.feat.compute(img_gray,kp)
        features = [Feature(pt23D(k.pt,getSubpix(depth,k),self.A),d) for k,d in zip(kp,desc) if getSubpix(depth,k) > 0]

        if self.prevImg is not None:
            # Match against previous and map
            prevMatch = match(self.prevFeat,features)
            mapMatch = match(self.Map.features,features)

            # Get geometry
            trPrev,matchPrev,featPrev = findTransform(self.prevFeat,features,prevMatch)
            trPrev = np.matmul(self.transform,trPrev)
            trMap,matchMap,featMap = findTransform(self.Map.features,features,mapMatch)

            # Run kalman filter
            self.transform = self.KF(trPrev,trMap)

            # Update features in map
            self.Map.updateFeatrues(featMap,matchMap,self.transform)

            # Add new features to map
            newFeat = [f for f in featPrev if f not in featMap]
            self.Map.addFeatures(newFeat)

            # Add to pc
            self.PC.update(img,depth,self.A,self.transform)

            # Visualize
        else:
            self.Map.addFeatures(features)


        self.prevImg = img
        self.prevDepth = depth
        self.prevFeat = features

    def run(self):
        i = 0
        for img,depth in self.Dataset:
            self.addFrame(img,depth)
            print("Frame %d" %i)
            i += 1
