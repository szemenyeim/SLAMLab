import cv2
import numpy as np
import math

def pt23D(center, depth, A):
    Z = depth*0.001
    X = Z * (center[0] - A[0, 2]) * A[0, 0]
    Y = -Z * (center[1] - A[1, 2]) * A[1, 1]
    return np.array([X, Y, Z])

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

def transformPoints(pts,mtx):
    ptsH = np.append(pts,np.ones((pts.shape[0],1)),1)
    tranH = np.transpose(np.matmul(mtx,np.transpose(ptsH)))
    return tranH[:,:3]

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

class RANSAC(object):
    def __init__(self,dThresh=0.1,N=5,mult=10):
        self.dThresh = dThresh
        self.N = N
        self.mult = mult

    def generateCandidate(self,src,dst):
        mtx = np.eye(4, 4)

        if len(src) == 0 or len(dst) == 0:
            return None

        srcCent = np.mean(src, 0)
        dstCent = np.mean(dst, 0)

        H = np.matmul(np.transpose(src - srcCent), (dst - dstCent))

        S, U, Vt = cv2.SVDecomp(H)
        R = np.matmul(U, Vt)
        if cv2.determinant(R) < 0:
            return None
        mtx[0:3, 0:3] = np.transpose(R)
        mtx[0:3, 3] = dstCent - np.matmul(np.transpose(R), srcCent)

        return mtx

    def evalCandidate(self,mtx,src,dst):
        trSrc = transformPoints(src, mtx)
        distances = [np.sum((s - d) ** 2) for s, d in zip(trSrc, dst)]
        inliers = [1 if d < self.dThresh else 0 for d in distances]
        return inliers

    def __call__(self, src,dst,matches=None):

        if matches is None:
            matches = [cv2.DMatch(i,i,0) for i in range(src.shape[0])]
            matchDst = [dst[m.trainIdx] for m in matches]
            srcCoords = src
            dstCoords = dst
        else:
            matchSrc = [src[m.queryIdx] for m in matches]
            matchDst = [dst[m.trainIdx] for m in matches]
            srcCoords = np.array([np.array(f.center) for f in matchSrc])
            dstCoords = np.array([np.array(f.center) for f in matchDst])

        n = srcCoords.shape[0]
        if n == 0:
            return None,[],[]
        N = min(400,max(2000,self.mult*n))

        candidates = []

        for i in range(N):
            ind = np.random.randint(0,n,self.N)
            c = self.generateCandidate(srcCoords[ind,:],dstCoords[ind,:])
            if c is not None:
                candidates.append(c)

        inliers = [self.evalCandidate(c,srcCoords,dstCoords) for c in candidates]
        scores = [sum(i) for i in inliers]

        best_i = np.argmax(scores)
        inliers = np.array(inliers[best_i],dtype='bool')
        mtx = self.generateCandidate(srcCoords[inliers],dstCoords[inliers])
        if mtx is None:
            mtx = candidates[best_i]

        goodMatches = [matches[i] for i in range(len(inliers)) if inliers[i] == 1]
        goodFeatures = [matchDst[i] for i in range(len(inliers)) if inliers[i] == 1]

        for i, m in enumerate(goodMatches):
            m.trainIdx = i

        return mtx, goodMatches, goodFeatures

class Kalman(object):
    def __init__(self,dt):
        self.KF = cv2.KalmanFilter(18,12)
        self.dt = dt
        cv2.setIdentity(self.KF.processNoiseCov,1e-5)
        cv2.setIdentity(self.KF.measurementNoiseCov,1e-4)
        cv2.setIdentity(self.KF.errorCovPost,1)

        # position
        transitionMatrix = np.eye(18)
        transitionMatrix[0, 3] = dt
        transitionMatrix[1, 4] = dt
        transitionMatrix[2, 5] = dt
        transitionMatrix[3, 6] = dt
        transitionMatrix[4, 7] = dt
        transitionMatrix[5, 8] = dt
        transitionMatrix[0, 6] = 0.5 * pow(dt, 2)
        transitionMatrix[1, 7] = 0.5 * pow(dt, 2)
        transitionMatrix[2, 8] = 0.5 * pow(dt, 2)
        # orientation
        transitionMatrix[9, 12] = dt
        transitionMatrix[10, 13] = dt
        transitionMatrix[11, 14] = dt
        transitionMatrix[12, 15] = dt
        transitionMatrix[13, 16] = dt
        transitionMatrix[14, 17] = dt
        transitionMatrix[9, 15] = 0.5 * pow(dt, 2)
        transitionMatrix[10, 16] = 0.5 * pow(dt, 2)
        transitionMatrix[11, 17] = 0.5 * pow(dt, 2)
        self.KF.transitionMatrix = transitionMatrix.astype('float32')

        measurementMatrix = np.zeros((12,18))
        measurementMatrix[0, 0] = 1
        measurementMatrix[1, 1] = 1
        measurementMatrix[2, 2] = 1
        measurementMatrix[3, 0] = 1
        measurementMatrix[4, 1] = 1
        measurementMatrix[5, 2] = 1
        measurementMatrix[6, 9] = 1
        measurementMatrix[7, 10] = 1
        measurementMatrix[8, 11] = 1
        measurementMatrix[9, 9] = 1
        measurementMatrix[10, 10] = 1
        measurementMatrix[11, 11] = 1
        self.KF.measurementMatrix = measurementMatrix.astype('float32')

    def getMeas(self,tr1,tr2):
        if tr2 is None:
            tr2 = tr1
        measured_eulers1 = rot2euler(tr1[0:3,0:3])
        measured_eulers2 = rot2euler(tr2[0:3,0:3])

        measurements = np.zeros(12)
        measurements[0] = tr1[0,3]
        measurements[1] = tr1[1,3]
        measurements[2] = tr1[2,3]
        measurements[3] = tr2[0,3]
        measurements[4] = tr2[1,3]
        measurements[5] = tr2[2,3]
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
        trOut[0,3] = estimate[0]
        trOut[1,3] = estimate[1]
        trOut[2,3] = estimate[2]

        return trOut