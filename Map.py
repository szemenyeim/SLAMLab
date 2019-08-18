import cv2
import numpy as np
import copy

class Map(object):
    def __init__(self):
        self.alpha = 0.2
        self.features = []

    def addFeatures(self,features,transform):
        if len(features) > 0:
            feat = copy.deepcopy(features)
            for f in feat:
                f.center = np.matmul(transform[0:3,0:3], np.array(f.center)) + transform[0:3,3]
            self.features.extend(feat)

    def updateFeatrues(self,features,matches,transform):
        if len(features) == 0 or len(matches) == 0:
            return

        # Transform features
        tFeatures = copy.deepcopy(features)
        for f in tFeatures:
            f.center = np.matmul(transform[0:3,0:3], np.array(f.center)) + transform[0:3,3]

        # Update coords and descriptors
        for m in matches:
            self.features[m.queryIdx] = self.alpha*tFeatures[m.trainIdx] + ((1-self.alpha)*self.features[m.queryIdx])