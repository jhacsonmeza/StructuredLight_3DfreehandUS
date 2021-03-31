import numpy as np
import glob
import cv2
import os

from sl_us import utils

from sl_us import centerline
from sl_us import sl3d
from sl_us import us3d


################################# Input data ##################################
# Paths of data
base = os.path.relpath('dataset')
root1 = os.path.join(base,'SLdata')
root2 = os.path.join(base,'USdata')


# Input images for structured light reconstruction
imlist_sl = utils.natsort(glob.glob(os.path.join(root1,'images','*')))
mask_sl = os.path.join(root1,'mask.png')

# Input images for freehand ultrasound reconstruction
imlist_us = utils.natsort(glob.glob(os.path.join(root2,'US','*')))
masks_us = utils.natsort(glob.glob(os.path.join(root2,'USmasks','*')))


# Load camera-projector stereo calibration parameters
Params1 = np.load(os.path.join(base,'cam1_proj_params.npz'))

# Load freehand ultrasound parameters
Params2 = np.load(os.path.join(base,'US_params.npz'))


# Phase-shifting steps and shifts for the structured light images
N = 8 # Total of steps used
p = 12 # Pitch/period of the fringes




################################ SL Reconstruction ############################
im = cv2.imread(imlist_sl[-1], 0)
mask = np.int32(cv2.imread(mask_sl, 0)/255)

# Estimate absolute phase in x using centreline image
phix = centerline.NStepPhaseShifting(imlist_sl[1:9], N)
p0 = centerline.seedPoint(imlist_sl[0], mask, True)
Phix = centerline.spatialUnwrap(phix, p0, mask)

# Establish correspondences
yc, xc = np.where(mask) # Camera 2D coordinates
xp = p/2/np.pi*Phix[yc,xc] + (1280-1)/2 # Projector x coordinates

# Triangulation
X1 = sl3d.triangulate(xc, yc, xp, Params1)
C1 = im[mask == 1]



############################## US Reconstruction ##############################
T_T_W = np.load(os.path.join(root2,'target_pose.npy'))

# 3D map B-scans
X2, C2 = us3d.mapMask(imlist_us, masks_us, T_T_W, Params2)



########################### Save multimodal results ###########################
utils.writePointCloud(np.hstack([X1,X2]), np.append(C1,C2),
                      os.path.join(base,'SL_US_3D.ply'))