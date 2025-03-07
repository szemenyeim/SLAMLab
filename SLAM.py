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
import platform

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
        self.feat = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_KAZE,threshold=0.0005)
        self.useMap = True
        np.random.seed(1)

    # One SLAM step
    def addFrame(self,img,depth):

        # Convert image to gray (create new)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Detect features
        keypoints = self.feat.detect(img_gray)

        # Remove no depth keypoints
        keypoints = [k for k in keypoints if getSubpix(depth,k) > 0]

        # Compute descriptors
        keypoints,descriptors = self.feat.compute(img_gray,keypoints)

        # Get features (we need 3D features only)
        features = [Feature(pt23D(k.pt,getSubpix(depth,k),self.A),d) for k,d in zip(keypoints,descriptors)]


        if self.prevImg is not None:

            # Match against previous
            prevMatch = match(self.prevFeat,features)

            # draw features
            draw = cv2.drawMatches(self.prevImg,self.prevKp,img,keypoints,prevMatch,None)
            cv2.imshow("matches",draw)
            cv2.imshow("img",img)
            cv2.imshow("depth",img*np.expand_dims(depth/10000,2)/255)
            cv2.waitKey(1)

            # Get relative transform
            trPrev,matchPrev,featPrev = self.RANSAC(self.prevFeat,features,prevMatch)
            # Get transform from the first frame
            trPrev = np.matmul(self.transform,trPrev)

            if self.useMap:
                # Match against map
                mapMatch = match(self.Map.features,features)
                # Get transform
                trMap,matchMap,featMap = self.RANSAC(self.Map.features,features,mapMatch)

            # Run kalman filter
            self.transform = trMap if self.useMap else trPrev#self.KF(trPrev,trMap)
            #print(self.KF.getMeas(self.transform,None)[[0,1,2,6,7,8]])

            if self.useMap:
                # Update features in map
                self.Map.updateFeatrues(featMap,matchMap,np.linalg.inv(self.transform))

                # Get new features (features in featPrev, but not in featMap)
                newFeat = [f for f in featPrev if f not in featMap]
                # Add new features
                self.Map.addFeatures(newFeat,np.linalg.inv(self.transform))

            # Update point cloud
            self.PC.update(img,depth,self.A,self.transform)

        else:
            if self.useMap:
                self.Map.addFeatures(features,self.transform)

        # Update prevoius values
        self.prevImg = img
        self.prevDepth = depth
        self.prevFeat = features
        self.prevKp = keypoints

    if True:
        def visualize(self,i):
            xyz = self.PC.pc[:, 0:3]
            rgb = self.PC.pc[:, 3:6]/255.0
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            o3d.visualization.draw_geometries([pcd])
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
