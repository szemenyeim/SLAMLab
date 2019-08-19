from Geometry import *

class Kalman(object):
    def __init__(self,dt):

        # Create KALMAN filter with 18 states (6 variables + first and second derivatives)
        # and 12 observations (2 independent observations of the 6 variables)
        self.KF = cv2.KalmanFilter(18,12)

        # Set time between frames
        self.dt = dt

        # Set noise covariance matrices
        cv2.setIdentity(self.KF.processNoiseCov,1e-5)
        cv2.setIdentity(self.KF.measurementNoiseCov,1e-4)
        cv2.setIdentity(self.KF.errorCovPost,1)

        # Set transition matrix (we are in discrete so identity is the base)
        transitionMatrix = np.eye(18)

        # Put dt to the right places
        transitionMatrix[0, 3] = dt
        transitionMatrix[1, 4] = dt
        transitionMatrix[2, 5] = dt
        transitionMatrix[3, 6] = dt
        transitionMatrix[4, 7] = dt
        transitionMatrix[5, 8] = dt
        transitionMatrix[9, 12] = dt
        transitionMatrix[10, 13] = dt
        transitionMatrix[11, 14] = dt
        transitionMatrix[12, 15] = dt
        transitionMatrix[13, 16] = dt
        transitionMatrix[14, 17] = dt

        # Put dt squared / 2 to the right places
        transitionMatrix[0, 6] = 0.5 * pow(dt, 2)
        transitionMatrix[1, 7] = 0.5 * pow(dt, 2)
        transitionMatrix[2, 8] = 0.5 * pow(dt, 2)
        transitionMatrix[9, 15] = 0.5 * pow(dt, 2)
        transitionMatrix[10, 16] = 0.5 * pow(dt, 2)
        transitionMatrix[11, 17] = 0.5 * pow(dt, 2)

        # Set transition matrix
        self.KF.transitionMatrix = transitionMatrix.astype('float32')

        # Set measurement matrix
        measurementMatrix = np.zeros((12,18))

        # We observe the 6 variables directly
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

        # Set measurement matrix
        self.KF.measurementMatrix = measurementMatrix.astype('float32')

    # Convert transformation matrices to measurements
    def getMeas(self,tr1,tr2):
        if tr2 is None:
            tr2 = tr1

        # Convert rotation matrices to Euler angles
        measured_eulers1 = rot2euler(tr1[0:3,0:3])
        measured_eulers2 = rot2euler(tr2[0:3,0:3])

        # Set measurements
        measurements = np.zeros(12)

        # Translations
        measurements[0] = tr1[0,3]
        measurements[1] = tr1[1,3]
        measurements[2] = tr1[2,3]
        measurements[3] = tr2[0,3]
        measurements[4] = tr2[1,3]
        measurements[5] = tr2[2,3]

        # Rotations
        measurements[6] = measured_eulers1[0]
        measurements[7] = measured_eulers1[1]
        measurements[8] = measured_eulers1[2]
        measurements[9] = measured_eulers2[0]
        measurements[10] = measured_eulers2[1]
        measurements[11] = measured_eulers2[2]

        return measurements.astype('float32')

    def __call__(self, tr1, tr2):

        # Run prediction step
        self.KF.predict()

        # Run measurement correction (call self.getMeas to get measurements)
        estimate = self.KF.correct(self.getMeas(tr1,tr2))

        # Initialize transformation matrix
        trOut = np.eye(4)

        # Convert Euler angles to Rotation matrix and set it
        trOut[0:3,0:3] = euler2rot([estimate[9],estimate[10],estimate[11]])

        # Put translations in the right places
        trOut[0,3] = estimate[0]
        trOut[1,3] = estimate[1]
        trOut[2,3] = estimate[2]

        return trOut