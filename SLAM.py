import cv2
import numpy as np
import os.path as osp
from Map import Map
from Geometry import *
from Feature import *
from Database import Dataset
from PointCloud import PointCloud
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
import pptk

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
        self.prevKp = None
        self.v = None
        self.RANSAC = RANSAC()
        self.PC = PointCloud()
        self.feat = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_KAZE,threshold=0.005)

    def addFrame(self,img,depth):
        # Detect features
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp = self.feat.detect(img_gray)
        kp,desc = self.feat.compute(img_gray,kp)
        features = [Feature(pt23D(k.pt,getSubpix(depth,k),self.A),d) for k,d in zip(kp,desc) if getSubpix(depth,k) > 0]

        if self.prevImg is not None:

            # Match against previous and get tranfsorm
            prevMatch = match(self.prevFeat,features)
            trPrev,matchPrev,featPrev = self.RANSAC(self.prevFeat,features,prevMatch)
            trPrev = np.matmul(self.transform,trPrev)

            # draw features
            draw = cv2.drawMatches(self.prevImg,self.prevKp,img,kp,prevMatch,None)
            cv2.imshow("matches",draw)
            cv2.imshow("img",img)
            cv2.imshow("depth",depth)
            cv2.waitKey(1)

            # Match against map and get transform
            mapMatch = match(self.Map.features,features)
            trMap,matchMap,featMap = self.RANSAC(self.Map.features,features,mapMatch)

            # Run kalman filter
            self.transform = self.KF(trPrev,trPrev)
            print(self.KF.getMeas(self.transform,None)[[0,1,2,6,7,8]])

            # Update features in map
            self.Map.updateFeatrues(featMap,matchMap,np.linalg.inv(self.transform))

            # Add new features to map
            newFeat = [f for f in featPrev if f not in featMap]
            self.Map.addFeatures(newFeat,np.linalg.inv(self.transform))

            # Add to pc
            self.PC.update(img,depth,self.A,self.transform)

        self.prevImg = img
        self.prevDepth = depth
        self.prevFeat = features
        self.prevKp = kp

    def visualize(self):
        if self.v is not None:
            self.v.clear()
            self.v.load(self.PC.pc[:,0:3],self.PC.pc[:,3:6]/255.0)
        else:
            self.v = pptk.viewer(self.PC.pc[:,0:3],self.PC.pc[:,3:6]/255.0)
        cv2.waitKey(0)

    def run(self):
        i = 0
        for img,depth in self.Dataset:
            self.addFrame(img,depth)
            #print("Frame %d" %i)
            # Visualize
            if i % 10 == 1:
                self.visualize()
            i += 1
