import cv2
import numpy as np
import math

def pt23D(center, depth, A):
    X = depth * (center[0] - A[0, 2]) / A[0, 0]
    Y = depth * (center[1] - A[1, 2]) / A[1, 1]
    return np.array([X, Y, depth])

def getSubpix(img,pt):
    
    x = int(pt.pt[0])
    y = int(pt.pt[1])

    x0 = cv2.borderInterpolate(x,   img.shape[1], cv2.BORDER_REFLECT_101)
    x1 = cv2.borderInterpolate(x+1, img.shape[1], cv2.BORDER_REFLECT_101)
    y0 = cv2.borderInterpolate(y,   img.shape[0], cv2.BORDER_REFLECT_101)
    y1 = cv2.borderInterpolate(y+1, img.shape[0], cv2.BORDER_REFLECT_101)

    a = pt.pt[0] - x
    c = pt.pt[1] - y

    d = (img[y0, x0] * (1.0 - a) + img[y0, x1] * a) * (1.0 - c) + \
        (img[y1, x0] * (1.0 - a) + img[y1, x1] * a) * c

    return d

def findTransform(src,dst,matches):

    mtx = np.eye(4,4)

    matchSrc = [src[m.queryIdx] for m in matches]
    matchDst = [dst[m.trainIdx] for m in matches]

    srcCoords =np.array([np.array(f.center) for f in matchSrc])
    dstCoords = np.array([np.array(f.center) for f in matchDst])

    _,tr,inliers = cv2.estimateAffine3D(srcCoords,dstCoords)

    mtx[0:3,:] = tr

    goodMatches = [matches[i] for i in range(inliers.shape[0]) if inliers[i] == 1]
    goodFeatures = [matchDst[i] for i in range(inliers.shape[0]) if inliers[i] == 1]

    '''U,S,Vt = cv2.SVDecomp(mtx[0:2,0:2])
    mtx[0:2, 0:2] = U*Vt'''

    return mtx, goodMatches, goodFeatures


def euler2rot(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def rot2euler(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

class Kalman(object):
    def __init__(self,dt):
        self.KF = cv2.KalmanFilter(18,12)
        self.dt = dt
        cv2.setIdentity(self.KF.processNoiseCov,1e-5)
        cv2.setIdentity(self.KF.measurementNoiseCov,1e-4)
        cv2.setIdentity(self.KF.errorCovPost,1)

        # position
        self.KF.transitionMatrix[0, 3] = dt
        self.KF.transitionMatrix[1, 4] = dt
        self.KF.transitionMatrix[2, 5] = dt
        self.KF.transitionMatrix[3, 6] = dt
        self.KF.transitionMatrix[4, 7] = dt
        self.KF.transitionMatrix[5, 8] = dt
        self.KF.transitionMatrix[0, 6] = 0.5 * pow(dt, 2)
        self.KF.transitionMatrix[1, 7] = 0.5 * pow(dt, 2)
        self.KF.transitionMatrix[2, 8] = 0.5 * pow(dt, 2)
        # orientation
        self.KF.transitionMatrix[9, 12] = dt
        self.KF.transitionMatrix[10, 13] = dt
        self.KF.transitionMatrix[11, 14] = dt
        self.KF.transitionMatrix[12, 15] = dt
        self.KF.transitionMatrix[13, 16] = dt
        self.KF.transitionMatrix[14, 17] = dt
        self.KF.transitionMatrix[9, 15] = 0.5 * pow(dt, 2)
        self.KF.transitionMatrix[10, 16] = 0.5 * pow(dt, 2)
        self.KF.transitionMatrix[11, 17] = 0.5 * pow(dt, 2)

        self.KF.measurementMatrix[0, 0] = 1
        self.KF.measurementMatrix[1, 1] = 1
        self.KF.measurementMatrix[2, 2] = 1
        self.KF.measurementMatrix[3, 0] = 1
        self.KF.measurementMatrix[4, 1] = 1
        self.KF.measurementMatrix[5, 2] = 1
        self.KF.measurementMatrix[6, 9] = 1
        self.KF.measurementMatrix[7, 10] = 1
        self.KF.measurementMatrix[8, 11] = 1
        self.KF.measurementMatrix[9, 9] = 1
        self.KF.measurementMatrix[10, 10] = 1
        self.KF.measurementMatrix[11, 11] = 1

    def getMeas(self,tr1,tr2):
        measured_eulers1 = rot2euler(tr1[0:3,0:3])
        measured_eulers2 = rot2euler(tr2[0:3,0:3])

        measurements = np.zeros(12)
        measurements[0] = tr1[0,2]
        measurements[1] = tr1[1,2]
        measurements[2] = tr1[2,2]
        measurements[3] = tr2[0,2]
        measurements[4] = tr2[1,2]
        measurements[5] = tr2[2,2]
        measurements[6] = measured_eulers1[0]
        measurements[7] = measured_eulers1[1]
        measurements[8] = measured_eulers1[2]
        measurements[9] = measured_eulers2[0]
        measurements[10] = measured_eulers2[1]
        measurements[11] = measured_eulers2[2]

        return measurements.astype('float32')

    def __call__(self, tr1, tr2):
        self.KF.predict()
        estimate = self.KF.correct(self.getMeas(tr1,tr2))

        trOut = np.eye(4)
        trOut[0:3,0:3] = euler2rot([estimate[9],estimate[10],estimate[11]])
        trOut[0,2] = estimate[0]
        trOut[1,2] = estimate[1]
        trOut[2,2] = estimate[2]

        return trOut