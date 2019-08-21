import SLAM
from Camera import *
import sys

if __name__ == '__main__':
    root = "D:/Datasets/RGBD" if sys.platform == 'win32' else "./RGBD/"
    slam = SLAM.SLAM(root)
    slam.run()

    '''cam = Camera()
    i = 0
    while True:
        img,depth = cam.getImages()
        cv2.imshow("img",img)
        cv2.imshow("depth",depth)
        c = cv2.waitKey(1)
        if c == 13:
            cv2.imwrite(root + "rgb/" + str(i).zfill(2) + ".png",img)
            cv2.imwrite(root + "depth/" + str(i).zfill(2) + ".png",depth)
            i = i+1
        elif c == 27:
            exit(0)'''