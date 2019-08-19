import SLAM
import sys

if __name__ == '__main__':
    root = "D:/Datasets/RGBD" if sys.platform == 'win32' else "./RGDB"
    slam = SLAM.SLAM(root)
    slam.run()