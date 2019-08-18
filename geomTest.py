import numpy as np
from Geometry import *

def generatePoints(N):
    return np.random.randn(N,3)

def applyTransform(points,R,t):
    trMtx = np.zeros((3,4))
    trMtx[0:3,0:3] = R
    trMtx[0:3,3] = t
    return transformPoints(points,trMtx)

if __name__ == "__main__":

    for i in range(500):
        N = np.random.randint(20,200)
        M = np.random.randint(N/10,N*0.75)
        pts = generatePoints(N)

        ransac = RANSAC()

        angles = np.deg2rad(np.array([np.random.randint(-45,45), np.random.randint(-45,45), np.random.randint(-45,45)]))
        R = euler2rot(angles)
        t = np.array([np.random.randint(-15,15),np.random.randint(-15,15),np.random.randint(-15,15)])

        trPts = applyTransform(pts,R,t) + np.random.randn(N,3)/10
        trPts[N-M:N,:] = generatePoints(M) + t

        trans,goodMatches,goodFeatures = ransac(pts,trPts)

        anglesO = rot2euler(trans[0:3,0:3])
        tD = np.linalg.norm(trans[0:3,3]-t)
        tA = np.linalg.norm(np.rad2deg(anglesO)-np.rad2deg(angles))
        tIn = np.abs(N-M-len(goodMatches))/(N-M)
        if tD < 0.2 and tA < 5 and tIn < 0.5:
            None
        else:
            print("Match bad %d"%i)
            print(tD,tA,tIn)
            print(trans[0:3,3],t)
            print(np.rad2deg(anglesO),np.rad2deg(angles))
            print(N-M,len(goodMatches))