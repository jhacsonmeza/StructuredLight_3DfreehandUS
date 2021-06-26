import re
import cv2
import torch
import numpy as np

def labels2scores(labels, cell_size=8):
    scores = torch.nn.functional.softmax(labels, dim=1)[:,:-1]
    N, _, Hc, Wc = scores.shape
    H, W = Hc*cell_size, Wc*cell_size
    scores = scores.permute(0, 2, 3, 1).reshape(N, Hc, Wc, cell_size, cell_size)
    scores = scores.permute(0, 1, 3, 2, 4).reshape(N, H, W)

    return scores

def max_pool(scores, th):
    return torch.nn.functional.max_pool2d(scores, kernel_size=th*2+1, stride=1, padding=th)

def simple_nms(scores, th: int):
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores,th)
    
    supp_mask = max_pool(max_mask.float(),th) > 0
    supp_scores = torch.where(supp_mask, zeros, scores)
    new_max_mask = supp_scores == max_pool(supp_scores,th)
    max_mask = max_mask | (new_max_mask & (~supp_mask))
    
    return torch.where(max_mask, scores, zeros)
   


def getPose(Xo, Xx, Xy):
    '''
    Function to compute pose of the target
    
    input:
        Xo: 3D coordinates of point in the target that represent the origin
        Xx: 3D coordinates of point in the target in direction of x-axis
        Xy: 3D coordinates of point in the target in direction of y-axis
        
    output:
        R: rotation matrix
        t: translation vector
    '''
    
    xaxis = Xx-Xo # Vector pointing to x direction
    xaxis = xaxis/np.linalg.norm(xaxis) # Conversion to unitary
    
    yaxis = Xy-Xo # Vector pointing to y direction
    yaxis = yaxis/np.linalg.norm(yaxis) # Conversion to unitary
    
    zaxis = np.cross(xaxis, yaxis) # Unitary vector pointing to z direction
    
    # Build rotation matrix and translation vector
    R = np.c_[xaxis,yaxis,zaxis]
    t = Xo
    
    return R, t 
    
def drawAxes(im, K1, dist1, R, t):
    '''
    Function to project and draw the axes of the target coordinate system in 
    the image plane 1
    
    input:
        im: image of camera 1
        K1: intrinsic matrix of camera 1
        dist1: distortion coefficients
        R, t: position and orientation of target frame relative to camera 1
        
    output:
        image with axes drawn
    '''
    axes = 40*np.array([[0,0,0], [1.,0,0], [0,1.,0], [0,0,1.]]) # axes to draw
    
    # Reproject target coordinate system axes
    rvec, _ = cv2.Rodrigues(R)
    axs, _ = cv2.projectPoints(axes, rvec, t, K1, dist1)
    axs = np.int32(axs[:,0,:])
    
    # Draw axes
    origin = tuple(axs[0])
    im = cv2.line(im, origin, tuple(axs[1]), (0,0,255), 5)
    im = cv2.line(im, origin, tuple(axs[2]), (0,255,0), 5)
    im = cv2.line(im, origin, tuple(axs[3]), (255,0,0), 5)
    
    return im
