import cv2
import numpy as np

class Map(object):
    def __init__(self):
        self.alpha = 0.2
        self.features = []

    def addFeatures(self,features):
        self.features.extend(features)

    def updateFeatrues(self,features,matches,transform):
        # Transform features
        for f in features:
            f.center = np.matmul(transform[0:3,0:3], np.array(f.center)) + transform[0:3,3]

        # Update coords and descriptors
        for m in matches:
            self.features[m.queryIdx] = self.alpha*features[m.trainIdx] + ((1-self.alpha)*self.features[m.queryIdx])