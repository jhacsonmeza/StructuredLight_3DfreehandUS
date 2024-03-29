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

# Based on https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superpoint.py
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



def sortedPoints(scores, labels):
    # Extract keypoints
    kp = torch.nonzero(scores > 0.015, as_tuple=False) #(n,2) with rows,cols

    # If less than 3 points, not sort and return
    if kp.shape[0] < 3: return kp


    # Keep the 3 keypoints with highest score
    if kp.shape[0] > 3:
        _, indices = torch.topk(scores[tuple(kp.t())], 3, dim=0)
        kp = kp[indices]

    # Class ID
    r,c = (kp//8).T
    id_class = labels[r,c]

    if torch.any(id_class == 3):
        id_class[id_class == 3] = 6-id_class.sum()

    # Sort keypoints
    kp = kp[torch.argsort(id_class)].cpu().numpy()


    # from (row,col) to (x,y)
    kp = np.fliplr(kp)

    # Using Shoelace formula to know orientation of point
    A = kp[1,0]*kp[2,1]-kp[2,0]*kp[1,1] - \
        kp[0,0]*kp[2,1]+kp[2,0]*kp[0,1] + \
        kp[0,0]*kp[1,1]-kp[1,0]*kp[0,1]
    
    # Correct orientation if needed
    if A > 0:
        kp = kp[[1,0,2]]

    return kp

def extractPatches(im1, im2, kp1, kp2, crop_sz, mid):
    # Estimate top-left point for both views
    tl1 = np.int32(np.round(kp1)) - mid
    tl2 = np.int32(np.round(kp2)) - mid

    # Create patches
    patches1 = []
    patches2 = []
    for i in range(3):
        imcrop1 = im1[tl1[i,1]:tl1[i,1]+crop_sz, tl1[i,0]:tl1[i,0]+crop_sz]
        imcrop2 = im2[tl2[i,1]:tl2[i,1]+crop_sz, tl2[i,0]:tl2[i,0]+crop_sz]
        
        # Check if the patch need to be outsied of the image
        if imcrop1.shape != (crop_sz,crop_sz): 
            imcrop1 = correct_patch(im1, kp1, tl1[i], crop_sz, mid)
        
        if imcrop2.shape != (crop_sz,crop_sz): 
            imcrop2 = correct_patch(im2, kp2, tl2[i], crop_sz, mid)
        
        # Save patches
        patches1.append(imcrop1)
        patches2.append(imcrop2)
    
    patches = np.stack(patches1+patches2, 0)

    return patches

def correct_patch(im, kp, tl, crop_sz, mid):
    union_xy = np.minimum(0,tl)
    union_wh = np.maximum(im.shape[::-1], tl+crop_sz) - union_xy

    cx = union_wh[0] - im.shape[1] + 1
    cy = union_wh[1] - im.shape[0] + 1
    if union_xy[0] == 0: cx = -cx
    if union_xy[1] == 0: cy = -cy

    # Transform image
    Ht = np.array([[1.,0.,cx],[0.,1.,cy]])
    im = cv2.warpAffine(im.copy(),Ht,None,None,cv2.INTER_LINEAR,cv2.BORDER_REPLICATE)

    # Modify tl corner
    c = kp.copy()
    c[0] += cx
    c[1] += cy
    tl = np.int32(np.round(c)) - mid

    # Crop
    imcrop = im[tl[1]:tl[1]+crop_sz, tl[0]:tl[0]+crop_sz]

    return imcrop

def ellipseFitting(masks):
    centers = []
    for mask in masks:
        contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        idx = 0
        if len(contours) > 1:
            areas = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                areas.append(area)
            idx = np.argmax(areas).item()
        
        rbox = cv2.fitEllipse(contours[idx])
        centers.append([rbox[0][0],rbox[0][1]])
    
    return np.array(centers)



def getPose(Xo, Xx, Xy):
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