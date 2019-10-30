import os.path as osp
from Map import Map
from Geometry import *
from Feature import *
from Database import Dataset
from PointCloud import PointCloud
from RANSAC import RANSAC
from Kalman import Kalman
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
import platform

rel = platform.linux_distribution() #returns release version

class SLAM(object):
    def __init__(self,root,fps = 1):
        self.frameCnt = 0
        self.Dataset = Dataset(root)
        self.Map = Map()
        self.A = np.genfromtxt("cam.txt", delimiter=',')
        self.fps = fps
        self.transform = np.eye(4,4)
        self.KF = Kalman(1.0/self.fps)
        self.prevImg = None
        self.prevDepth = None
        self.prevFeat = None
        self.prevKp = None
        self.v = None
        self.RANSAC = RANSAC()
        self.PC = PointCloud()
        np.random.seed(1)
        #TODO: create feature detector

    # One SLAM step
    def addFrame(self,img,depth):

        #TODO: Convert image to gray (create new)

        #TODO: Detect features
        keypoints = []

        #TODO: Remove no depth keypoints

        #TODO: Compute descriptors

        #TODO: Construct list of features (don't forget to get the 3D coordinate of the feature)
        features = []

        if self.prevImg is not None:

            #TODO: Match against previous

            # Draw features
            #TODO: Uncomment
            '''draw = cv2.drawMatches(self.prevImg,self.prevKp,img,keypoints,prevMatch,None)
            cv2.imshow("matches",draw)'''
            cv2.imshow("img",img)
            cv2.imshow("depth",img*np.expand_dims(depth/10000,2)/255)
            cv2.waitKey(1)

            #TODO: Get relative transform

            #TODO: Get absolute transform


            #TODO: Match against map

            #TODO: Get transform


            #TODO: Run kalman filter


            #TODO: Update features in map


            #TODO: Get new features (features in featPrev, but not in featMap)

            #TODO: Add new features


            # Update point cloud
            self.PC.update(img,depth,self.A,self.transform)

        # Update prevoius values
        self.prevImg = img
        self.prevDepth = depth
        self.prevFeat = features
        self.prevKp = keypoints

    if rel[1] == '18.04':
        def visualize(self,i):
            import pptk
            # If visualizer already exists
            if self.v is not None:
                # Clear and load
                self.v.clear()
                self.v.load(self.PC.pc[:,0:3],self.PC.pc[:,3:6]/255.0)
            else:
                # Create new one
                self.v = pptk.viewer(self.PC.pc[:,0:3],self.PC.pc[:,3:6]/255.0)
            cv2.waitKey(0)
    else:
        def visualize(self, i):
            from pypcd import pypcd
            rgb = np.expand_dims(pypcd.encode_rgb_for_pcl(self.PC.pc[:,3:6].astype('uint8')),1)
            pc = np.hstack([self.PC.pc[:,0:3],rgb]).astype('float32')
            pc = pypcd.make_xyz_rgb_point_cloud(pc)
            pc.save_pcd("Clouds/%dCloud.pcd" % i, compression='ascii')
            cv2.waitKey(0)

    def run(self):

        for i, (img,depth) in enumerate(self.Dataset):

            self.addFrame(img,depth)

            # Visualize
            self.visualize(i)
