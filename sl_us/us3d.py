import numpy as np
import cv2

def mapBscans(imlist, T_T_W, params):
    C = np.array([cv2.imread(im,0).flatten() for im in imlist]).flatten()
    
    bscan = cv2.imread(imlist[0], 0)
    y, x = np.where(bscan**0)
    zeros = np.zeros(x.shape)
    pts = np.c_[params['sx']*x, params['sy']*y, zeros, zeros+1].T

    X = []
    for T_T_Wi in T_T_W:
        X.append(T_T_Wi @ params['T_I_T'] @ pts)
    X = np.hstack(X)[:3]
        
    return X, C

def mapMask(imlist, masks, T_T_W, params):
    sx, sy = params['sx'], params['sy']
    T_I_T = params['T_I_T']

    X = []
    C = np.array([], np.uint8)
    for i in range(len(imlist)):
        mask = cv2.imread(masks[i],0)

        y, x = np.where(mask)
        zeros = np.zeros(x.shape)
        pts = np.c_[sx*x, sy*y, zeros, zeros+1].T

        # Map image points to 3D space
        X.append(T_T_W[i] @ T_I_T @ pts)
        # Save color
        im = cv2.imread(imlist[i],0)
        C = np.append(C, im[y,x])

    X = np.hstack(X)[:3]
    return X, C


def mapPoints(imlist, pointlist, T_T_W, params):
    X = []
    C = np.array([], np.uint8)
    for i in range(len(imlist)):
        x, y = pointlist[i].T
        zeros = np.zeros(x.shape)
        pts = np.c_[params['sx']*x, params['sy']*y, zeros, zeros+1].T
        X.append(T_T_W[i] @ params['T_I_T'] @ pts)
        
        im = cv2.imread(imlist[i],0)
        C = np.append(C, im[y,x])
        
    X = np.hstack(X)[:3]
        
    return X, C