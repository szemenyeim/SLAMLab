import numpy as np
import cv2
import os
import os.path as osp
import glob

class Dataset(object):
    def __init__(self,root):
        self.root = root
        self.imgRoot = osp.join(root,"rgb")
        self.depthRoot = osp.join(root,"depth")
        self.images = sorted(glob.glob1(self.imgRoot,"*.png"))
        self.depths = sorted(glob.glob1(self.depthRoot,"*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):

        img = cv2.imread(osp.join(self.imgRoot,self.images[i]))
        depth = cv2.imread(osp.join(self.depthRoot,self.depths[i]),-1)

        return img,depth
