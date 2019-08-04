import cv2
import numpy as np
from Geometry import *

class PointCloud(object):
    def __init__(self):
        self.pc = np.ndarray((0,6))

    def update(self,img,depth,A,tr):

        img_dow = cv2.resize(img,(320,240))
        depth_dow = cv2.resize(depth,(320,240))

        ptCnt = np.count_nonzero(depth_dow)
        points = np.ndarray((ptCnt,6))
        k = 0

        for i in range(img_dow.shape[0]):
            for j in range(img_dow.shape[1]):
                color = img_dow[i,j,:]
                d = depth_dow[i,j]
                if d > 0:
                    pt = np.concatenate((pt23D((i,j),d,A),color))
                    points[k,:] = np.array(pt)
                    k += 1

        points[:,0:3] = transformPoints(points[:,0:3],np.linalg.inv(tr))
        self.pc = np.concatenate((self.pc,points))