import os.path as osp
from Map import Map
from Geometry import *
from Feature import *
from Database import Dataset
from PointCloud import PointCloud
from RANSAC import RANSAC
import open3d as o3d
#from Kalman import Kalman
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

class SLAM(object):
    def __init__(self,root,fps = 1):
        self.frameCnt = 0
        self.Dataset = Dataset(root)
        self.Map = Map()
        self.A = np.genfromtxt("cam.txt", delimiter=',')
        self.fps = fps
        self.transform = np.eye(4,4)
        #self.KF = Kalman(1.0/self.fps)
        self.prevImg = None
        self.prevDepth = None
        self.prevFeat = None
        self.prevKp = None
        self.v = None
        self.RANSAC = RANSAC()
        self.PC = PointCloud()
        self.useMap = False
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

            # TODO: Match against previous

            # Draw features
            # TODO: Uncomment
            '''draw = cv2.drawMatches(self.prevImg,self.prevKp,img,keypoints,prevMatch,None)
            cv2_imshow(draw)
            cv2.waitKey(1)'''

            # TODO: Get relative transform

            # TODO: Extra task only
            if self.useMap:
                pass
                # Match against map

                # Get transform

            # TODO: Get absolute transform

            # TODO: Extra task only
            if self.useMap:
                pass
                # Update features in map

                # Get new features (features in featPrev, but not in featMap)

                # Add new features

            # Update point cloud
            self.PC.update(img, depth, self.A, self.transform)

        else:
            if self.useMap:
                # Initialize map
                self.Map.addFeatures(features, self.transform)

        # Update prevoius values
        self.prevImg = img
        self.prevDepth = depth
        self.prevFeat = features
        self.prevKp = keypoints

    def visualize(self,i):
        xyz = self.PC.pc[:, 0:3]
        rgb = self.PC.pc[:, 3:6]/255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.visualization.draw_geometries([pcd])

    def run(self):

        for i, (img,depth) in enumerate(self.Dataset):

            self.addFrame(img,depth)

        # Visualize
        self.visualize(i)
