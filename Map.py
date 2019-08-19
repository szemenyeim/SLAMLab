import cv2
import numpy as np
import copy

# Feature map to do absolute positioning with
class Map(object):
    def __init__(self):
        self.alpha = 0.2
        self.features = []

    # Add new feature to the map
    def addFeatures(self,features,transform):
        if len(features) > 0:
            # create deep copy (avoid corrupting features for the next loop)
            feat = copy.deepcopy(features)

            # Transform feature positions
            for f in feat:
                f.center = np.matmul(transform[0:3,0:3], np.array(f.center)) + transform[0:3,3]

            # Add new features
            self.features.extend(feat)

    # Update existing feature positions and descriptors
    def updateFeatrues(self,features,matches,transform):
        if len(features) == 0 or len(matches) == 0:
            return

        # create deep copy (avoid corrupting features for the next loop)
        tFeatures = copy.deepcopy(features)

        # Transform features
        for f in tFeatures:
            f.center = np.matmul(transform[0:3,0:3], np.array(f.center)) + transform[0:3,3]

        # Update coords and descriptors
        for m in matches:
            self.features[m.queryIdx] = self.alpha*tFeatures[m.trainIdx] + ((1-self.alpha)*self.features[m.queryIdx])