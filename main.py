import SLAM
from Camera import *
import sys
import os

if __name__ == '__main__':

    useCam = False

    if useCam:
        cam = Camera()
        i = 0
        root = './Temp/'
        os.mkdir(root)
        os.mkdir(root + "rgb/")
        os.mkdir(root + "depth/")
        while True:
            img, depth = cam.getImages()
            cv2.imshow("img", img)
            cv2.imshow("depth", depth)
            c = cv2.waitKey(1)
            if c == 13:
                cv2.imwrite(root + "rgb/" + str(i).zfill(2) + ".png", img)
                cv2.imwrite(root + "depth/" + str(i).zfill(2) + ".png", depth)
                i = i + 1
            elif c == 27:
                break
    else:
        root = "./RGBD/"

    slam = SLAM.SLAM(root)
    slam.run()

