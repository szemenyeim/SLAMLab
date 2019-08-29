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

    #TODO: Create Flann based matcher

    #TODO: kNNMatch k=2

    # Create new list for good matches
    ratioGood = []

    #TODO: If best is considerably better than second best, then add it to good matches


    # Return good matches
    return ratioGood

