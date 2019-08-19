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
import pptk

class SLAM(object):
    def __init__(self,root,fps = 25):
        self.frameCnt = 0
        self.Dataset = Dataset(root)
        self.Map = Map()
        self.A = np.genfromtxt(osp.join(root, "cam.txt"), delimiter=',')
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
        self.feat = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_KAZE,threshold=0.005)

    # One SLAM step
    def addFrame(self,img,depth):

        # Conver image to gray (create new)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Detect features
        kp = self.feat.detect(img_gray)

        # Compute descriptors
        kp,desc = self.feat.compute(img_gray,kp)

        # Get features if the depth is not zero at their location (we need 3D features only)
        features = [Feature(pt23D(k.pt,getSubpix(depth,k),self.A),d) for k,d in zip(kp,desc) if getSubpix(depth,k) > 0]

        if self.prevImg is not None:

            # Match against previous
            prevMatch = match(self.prevFeat,features)
            # Get relative transform
            trPrev,matchPrev,featPrev = self.RANSAC(self.prevFeat,features,prevMatch)
            # Get transform from the first frame
            trPrev = np.matmul(self.transform,trPrev)

            # draw features
            draw = cv2.drawMatches(self.prevImg,self.prevKp,img,kp,prevMatch,None)
            cv2.imshow("matches",draw)
            cv2.imshow("img",img)
            cv2.imshow("depth",depth)
            cv2.waitKey(1)

            # Match against map
            mapMatch = match(self.Map.features,features)
            # Get transform
            trMap,matchMap,featMap = self.RANSAC(self.Map.features,features,mapMatch)

            # Run kalman filter
            self.transform = self.KF(trPrev,trPrev)
            #print(self.KF.getMeas(self.transform,None)[[0,1,2,6,7,8]])

            # Update features in map
            self.Map.updateFeatrues(featMap,matchMap,np.linalg.inv(self.transform))

            # Get new features (features in featPrev, but not in featMap)
            newFeat = [f for f in featPrev if f not in featMap]
            # Add new features
            self.Map.addFeatures(newFeat,np.linalg.inv(self.transform))

            # Update point cloud
            self.PC.update(img,depth,self.A,self.transform)

        # Update prevoius values
        self.prevImg = img
        self.prevDepth = depth
        self.prevFeat = features
        self.prevKp = kp

    def visualize(self):
        # If visualizer already exists
        if self.v is not None:
            # Clear and load
            self.v.clear()
            self.v.load(self.PC.pc[:,0:3],self.PC.pc[:,3:6]/255.0)
        else:
            # Create new one
            self.v = pptk.viewer(self.PC.pc[:,0:3],self.PC.pc[:,3:6]/255.0)
        cv2.waitKey(0)

    def run(self):

        for i, (img,depth) in enumerate(self.Dataset):

            self.addFrame(img,depth)

            # Visualize every 10 frames
            if i % 10 == 1:
                self.visualize()
