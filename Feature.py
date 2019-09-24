import cv2
import numpy as np

# Class for storing a feautre
class Feature(object):
    def __init__(self,center,descriptor):
        self.center = center
        self.descriptor = descriptor

    # Overload + operator
    def __add__(self, other):
        return Feature(self.center + other.center, self.descriptor + other.descriptor)

    # Overload * operator
    def __mul__(self, other):
        return Feature(np.array([c * other for c in self.center]), self.descriptor * other)

    # * operator with real numbers
    __rmul__ = __mul__

    # Overload == operator
    def __eq__(self, other):
        return (self.center == other.center).all() and self.descriptor == other.descriptor

# Match function
def match(src,dst):

    # If one of them is empty, return an empty list
    if len(dst) == 0 or len(src) == 0:
        return []

    # Get descriptors (convert to numpy array)
    dSrc = np.array([f.descriptor for f in src])
    dDst = np.array([f.descriptor for f in dst])

    # Create Flann based matcher
    matcher = cv2.FlannBasedMatcher()

    # kNNMatch k=2
    matches = matcher.knnMatch(dSrc,dDst,k=2)

    # Create new list for good matches
    ratioGood = [m for m,n in matches if m.distance/n.distance < 0.6]

    # Return good matches
    return ratioGood

