import numpy as np
import cv2


def estimateYProjectorCoordinate(ptsc, xp, params, undistort):
    if undistort:
        ptsc = _undistort_points(ptsc, params['K1'], params['dist1'], False)
    
    # Estimate epipolar line in the projector DMD
    lp = cv2.computeCorrespondEpilines(ptsc, 1, params['F'])

    # From ax+by+c=0 estimate yp coordinate as yp = -(ax+c)/b
    yp = -(lp[:,0,0]*xp+lp[:,0,2])/lp[:,0,1]
    
    return yp

def _undistort_points(pts, K, dist, normalize):
    if normalize:
        # Undistord and normalize coordinates
        ptsu = cv2.undistortPoints(pts, K, dist)
    else:
        # Only undistord points
        ptsu = cv2.undistortPoints(pts, K, dist, None, None, K)
    
    return ptsu

def PlaneLine(ptsc, ptsp, params, undistort):
    '''
    Triangulate using an inhomogeneous method where we solve for (X, Y, Z) in
    Euclidian coordinates and not in a projective space. This method is 
    equivalent to line and plane triangulation. Just the x or y coordinates
    of the projector are used.
    
    input:
        ptsc: N x 1 x 2 array with the cemera pixel coordinates
        ptsp: N x 1 x 2 array with the projector pixel coordinates
        params: stereo calibration parameters
    output:
        3 x N array with reconstructed 3D coordinates
    '''
    K1 = params['K1']
    K2 = params['K2']
    R = params['R']
    t = params['t']
    
    if undistort:
        # Undistort and normalize camera points
        ptsc = _undistort_points(ptsc, K1, params['dist1'], True)
        xc, yc = ptsc[:,0,:].T
        
        # Undistort and normalize projector points
        ptsp = _undistort_points(ptsp, K2, params['dist2'], True)
        xp, _ = ptsp[:,0,:].T
        
    else:
        # Convert camera and projector points from Euclidean to homogeneous
        ptsc = cv2.convertPointsToHomogeneous(ptsc)
        ptsp = cv2.convertPointsToHomogeneous(ptsp)
        
        # Normalize camera and projector points
        xc, yc, _ = np.linalg.inv(K1) @ ptsc[:,0,:].T
        xp, *_ = np.linalg.inv(K2) @ ptsp[:,0,:].T
    
    # Estimate 3D coordinates
    z = (-t[0,0]+t[2,0]*xp)/((R[0,0]-R[2,0]*xp)*xc + \
         (R[0,1]-R[2,1]*xp)*yc + R[0,2]-R[2,2]*xp)
    
    return np.array([z*xc, z*yc, z])

def triangulate(xc, yc, xp, params, undistort=True):
    # Camera 2D coordinates matrix
    ptsc = np.concatenate([xc[:,None], yc[:,None]], 1).astype(np.float64).reshape(-1,1,2)

    # Estimate y projector coordinates with epipolar line
    yp = estimateYProjectorCoordinate(ptsc, xp, params, undistort)
    # Projector 2D coordinates matrix
    ptsp = np.concatenate([xp[:,None], yp[:,None]], 1).reshape(-1,1,2)

    # Triangulate without y projector coordinates
    X = PlaneLine(ptsc, ptsp, params, undistort)
        
    return X