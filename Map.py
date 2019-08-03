import cv2
import numpy as np

class Map(object):
    def __init__(self):
        self.alpha = 0.2
        self.features = []

    def addFeatures(self,features):
        self.features.append(features)

    def updateFeatrues(self,features,matches,transform):
        print("a")
        # Transform features
        for f in features:
            f.coords = transform * f.coords

        # Update coords and descriptors
        for m in matches:
            self.features[m.queryIdx] = self.alpha*features[m.trainIdx] + ((1-self.alpha)*self.features[m.queryIdx])