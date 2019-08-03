import cv2
import numpy as np
import pcl
from Geometry import *

class PointCloud(object):
    def __init__(self):
        self.pc = pcl.PointCloud()

    def update(self,img,depth,A,tr):

        points = []

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                color = img[i,j,:]
                d = depth[i,j]
                if d > 0:
                    pt = pt23D((i,j),d,A)
                    points.append([pt,color])

        points = pcl.transformPointCloud(pcl.PointCloud(np.array(points)),tr)

        self.pc.append(points)
        filt = self.pc.make_voxel_filter()
        filt.set_leaf_size(0.1)
        self.pc = filt.filter()