import cv2
import numpy as np
import math

# 2D+depth -> 3D conversion
def pt23D(center, depth, A):
    #TODO: Convert Z to meters
    Z = 0
    #TODO: Compute X and Y
    X = 0
    Y = 0

    return np.array([X, Y, Z])

# Transform an array of points using homogeneous coordinates
def transformPoints(pts,mtx):
    #TODO: Add fourth coordinate (1)

    #TODO: Perform matrix multiplication (Don1t forget: both the input and the final output have to be transposed)

    #TODO: Return first 3 coordinates (assuming euclidean transforms)
    return None

# Get pixel value using bilinear interpolation
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

# Convert rotation matrix to euler angles
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

# Convert euler angles to rotation matrices
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