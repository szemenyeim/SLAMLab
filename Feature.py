import cv2
import numpy as np

class Feature(object):
    def __init__(self,center,descriptor):
        self.center = center
        self.descriptor = descriptor

    def __add__(self, other):
        return Feature(self.center + other.center, self.descriptor + other.descriptor)

    def __mul__(self, other):
        return Feature(np.array([c * other for c in self.center]), self.descriptor * other)

    __rmul__ = __mul__
    def __eq__(self, other):
        return (self.center == other.center).all() and self.descriptor == other.descriptor

def match(src,dst):

    dSrc = np.array([f.descriptor for f in src],dtype='float32')
    dDst = np.array([f.descriptor for f in dst],dtype='float32')

    matcher = cv2.FlannBasedMatcher()
    matcher.add(dDst)
    matcher.train()
    matches = matcher.knnMatch(dSrc,k=2)
    symGood = []

    for m,n in matches:
        if m.distance < 0.8*n.distance:
            symGood.append(m)

    return symGood

