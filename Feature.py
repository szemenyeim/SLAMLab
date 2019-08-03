import cv2
import numpy as np

class Feature(object):
    def __init__(self,center,descriptor):
        self.center = center
        self.descriptor = descriptor

    def __add__(self, other):
        return Feature(self.center + other.center, self.descriptor + other.descriptor)

    def __mul__(self, other):
        return Feature(self.center * other, self.descriptor * other)

    def __eq__(self, other):
        return self.center == other.center and self.descriptor == other.descriptor

def match(src,dst):

    dSrc = [f.desctriptor for f in src]
    dDst = [f.desctriptor for f in dst]

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(dSrc,dDst,k=2)
    symGood = []

    for m,n in matches:
        if m.distance < 0.8*n.distance:
            symGood.append([m])

    return symGood

