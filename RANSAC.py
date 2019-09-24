from Geometry import *


# RANSAC Algorithm
class RANSAC(object):
    def __init__(self,dThresh=0.001,N=4,mult=10):

        # Distance threshold for inliers
        self.dThresh = dThresh

        # Number of matches to use for candidate construction
        self.N = N

        # Multiplier for the number of candidates to generate
        self.mult = mult

    def generateCandidate(self,src,dst):

        # Initialize matrix
        mtx = np.eye(4, 4)

        # Check empty inputs (just in case)
        if len(src) == 0 or len(dst) == 0:
            return None

        # Compute centroids
        srcCent = np.mean(src, 0)
        dstCent = np.mean(dst, 0)

        # Compute H matrix
        H = np.matmul(np.transpose(src - srcCent), (dst - dstCent))

        # Get SVD
        S, U, Vt = cv2.SVDecomp(H)

        # R = U*Vt
        R = np.matmul(U, Vt)

        # If R is mirroring instead of Rotation, discard the candidate
        if cv2.determinant(R) < 0:
            return None

        # Put R in the transform
        mtx[0:3, 0:3] = np.transpose(R)

        # Compute translation
        mtx[0:3, 3] = dstCent - np.matmul(np.transpose(R), srcCent)

        return mtx

    # Get the inliers for a candidate
    def evalCandidate(self,mtx,src,dst):

        # Transform the src points using the candidate
        trSrc = transformPoints(src, mtx)

        # Compute distances
        distances = [np.sum((s - d) ** 2) for s, d in zip(trSrc, dst)]

        # get inliers
        inliers = [1 if d < self.dThresh else 0 for d in distances]
        return inliers

    # Run RANSAC (operator ())
    def __call__(self, src,dst,matches=None):

        if matches is None:
            # If matches is none, the algorithm's input are matching coordinates
            matches = [cv2.DMatch(i,i,0) for i in range(src.shape[0])]
            matchDst = [dst[m.trainIdx] for m in matches]
            srcCoords = src
            dstCoords = dst
        else:
            # Otherwise it's full features and matches
            matchSrc = [src[m.queryIdx] for m in matches]
            matchDst = [dst[m.trainIdx] for m in matches]
            srcCoords = np.array([np.array(f.center) for f in matchSrc])
            dstCoords = np.array([np.array(f.center) for f in matchDst])

        # Number of matching features. If 0, return none
        numMatches = srcCoords.shape[0]
        if numMatches == 0:
            return None,[],[]

        # Number of candidates to generate is 400 < self.mult*numMatches < 2000
        numCandidates = min(500,max(2000,self.mult*numMatches))

        candidates = []

        #TODO: Create indices array

        #TODO: Generate candidates
        candidates = []

        #TODO: Remove candidates that are None

        #TODO: get list of inliers for every candidate
        arrayOfInliers = []

        #TODO: Get number of inliers for every candidate
        scores = []

        #TODO: Get best candidate index

        #TODO: Get inlier list for best candidate and convert to bool
        inliers = []

        #TODO: Regenerate best candidate using all the inliers
        mtx = []

        #TODO: If the result is None, just use the best original candidate

        # get good Matches and good Features
        goodMatches = [matches[i] for i in range(len(inliers)) if inliers[i] == 1]
        goodFeatures = [matchDst[i] for i in range(len(inliers)) if inliers[i] == 1]

        # Change the train index in goodMatches to reflect the new list
        for i, m in enumerate(goodMatches):
            m.trainIdx = i

        return mtx, goodMatches, goodFeatures