from Geometry import *

# Class for point cloud
class PointCloud(object):

    def __init__(self):
        # XYZRGB points
        self.pc = np.ndarray((0,6))

    # Add new points
    def update(self,img,depth,A,tr):

        # Downscale image (avoid too many points in the cloud)
        img_dow = cv2.resize(img,(160,120))
        depth_dow = cv2.resize(depth,(160,120))

        # Count non-zero depth pixels (number of valid points)
        ptCnt = np.count_nonzero(depth_dow)
        points = np.ndarray((ptCnt,6))
        k = 0

        # Iterate through image
        for i in range(img_dow.shape[0]):
            for j in range(img_dow.shape[1]):

                # Get color and depth values
                color = img_dow[i,j,:]
                d = depth_dow[i,j]

                # If valid, add
                if d > 0:
                    # Compute 3D coordinates (use i*2 and j*2 because of downscaling by a factor of 2)
                    # and andd the pixel color to it
                    pt = np.concatenate((pt23D((j*4,i*4),d,A),color))

                    # Set point value
                    points[k,:] = np.array(pt)
                    k += 1

        # Tranform points using the estimated transform
        points[:,0:3] = transformPoints(points[:,0:3],np.linalg.inv(tr))

        # Concat
        self.pc = np.concatenate((self.pc,points))