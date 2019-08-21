import numpy as np
import cv2
from primesense import openni2
from primesense import _openni2 as c_api
import os.path as osp
from glob import glob1
import re
import freenect

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

class Camera(object):

    def getImages(self):
        img, _ = freenect.sync_get_video()
        depth, _ = freenect.sync_get_depth()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img, depth