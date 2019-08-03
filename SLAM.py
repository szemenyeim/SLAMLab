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
        self.A = np.genfromtxt(osp.join(root, "cam.csv"), delimiter=',')
        self.fps = fps
        self.transform = np.eye(4,4)
        self.KF = Kalman(1.0/self.fps)
        self.prevImg =None
        self.prevDepth = None
        self.prevFeat = None
        self.PC = PointCloud()
        self.feat = cv2.xfeatures2d_SURF.create(hessianThreshold=400)

    def addFrame(self,img,depth):
        # Detect features
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp = self.feat.detect(img_gray)
        desc = self.feat.compute(img_gray,kp)
        features = [Feature(pt23D(k,getSubpix(depth,k),self.A),d) for k,d in zip(kp,desc)]

        if self.prevImg:
            # Match against previous and map
            prevMatch = match(self.prevFeat,features)
            mapMatch = match(self.Map.features,features)

            # Get geometry
            trPrev,matchPrev,featPrev = findTransform(self.prevFeat,features,prevMatch)
            trPrev = self.transform*trPrev
            trMap,matchMap,featMap = findTransform(self.Map.features,features,mapMatch)

            # Run kalman filter
            trKalman = self.KF.getMeas(trPrev,trMap)
            self.transform[0:3,:] = trKalman

            # Update features in map
            self.Map.updateFeatrues(featMap,matchMap,trKalman)

            # Add new features to map
            newFeat = [f for f in featPrev if f not in featMap]
            self.Map.addFeatures(newFeat)

            # Add to pc
            self.PC.update(img,depth,self.A,trKalman)

            # Visualize


        self.prevImg = img
        self.prevDepth = depth
        self.prevFeat = features

    def run(self):
        for img,depth in self.Dataset:
            self.addFrame(img,depth)
