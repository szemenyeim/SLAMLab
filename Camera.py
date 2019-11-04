import numpy as np
import cv2
from primesense import openni2
from primesense import _openni2 as c_api
import os.path as osp
from glob import glob1
import re

type = False

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

    if type:
        def getImages(self):
            import freenect
            img, _ = freenect.sync_get_video()
            depth, _ = freenect.sync_get_depth(0,freenect.DEPTH_REGISTERED)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            return img, depth

    else:
        def __init__(self):
            openni2.initialize("/home/labor/Downloads/OpenNI/Linux/OpenNI-Linux-x64-2.3.0.63/Redist")
            self.dev = openni2.Device.open_any()
            self.depth_stream = self.dev.create_depth_stream()
            self.color_stream = self.dev.create_color_stream()
            self.dev.set_depth_color_sync_enabled( True )
            self.depth_stream.set_video_mode(
                c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=640,
                                   resolutionY=480, fps=30))
            self.depth_stream.start()
            self.color_stream.set_video_mode(
                c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640,
                                   resolutionY=480, fps=30))
            self.color_stream.start()
            self.dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

        def getImages(self):
            dframe = self.depth_stream.read_frame()
            dframe_data = dframe.get_buffer_as_uint16()
            depth = np.frombuffer(dframe_data, dtype=np.uint16)
            depth.shape = (480, 640, 1)

            cframe = self.color_stream.read_frame()
            cframe_data = cframe.get_buffer_as_uint8()
            img = np.frombuffer(cframe_data, dtype=np.uint8)
            img.shape = (480, 640, 3)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            return img, depth