import os
import cv2
import torch
import glob
import numpy as np

from sl_us.utils import natsort
from MarkerPose import models
from MarkerPose import utils


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Root path
root = os.path.abspath('dataset')

# Load stereo calibration parameters
Params = np.load(os.path.join(root, 'cam1_cam2_params.npz'))
K1 = Params['K1']
dist1 = Params['dist1']

# Create SuperPoint model
superpoint = models.SuperPointNet(3)
superpoint.load_state_dict(torch.load(os.path.join(root, 'superpoint.pt'), map_location=device))

# Create EllipSegNet model
ellipsegnet = models.EllipSegNet(16, 1)
ellipsegnet.load_state_dict(torch.load(os.path.join(root, 'ellipsegnet.pt'), map_location=device))

# Create MarkerPose model
markerpose = models.MarkerPose(superpoint, ellipsegnet, (320,240), 120, Params)
markerpose.to(device)
markerpose.eval()



# Set window name and size
cv2.namedWindow('Pose estimaion', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pose estimaion', 1792, 717)

# Read left and right images of the target
I1 = natsort(glob.glob(os.path.join(root, 'USdata', 'L', '*')))
I2 = natsort(glob.glob(os.path.join(root, 'USdata', 'R', '*')))


T_T_W = [] # Transformation from target frame {T} to world frame {W}
for im1n, im2n in zip(I1,I2):
    im1 = cv2.imread(im1n)
    im2 = cv2.imread(im2n)

    # Pose estimation
    R_T_W, t_T_W = markerpose(im1, im2)
    
    # Save pose
    T_T_W.append(np.r_[np.c_[R_T_W, t_T_W], [[0,0,0,1]]])
    
    # Visualize results
    utils.drawAxes(im1, K1, dist1, R_T_W, t_T_W)
    
    cv2.imshow('Pose estimaion',np.hstack([im1,im2]))
    if cv2.waitKey(255) == 27:
        break

cv2.destroyAllWindows()


# Save variables
np.save(os.path.join(root, 'USdata', 'target_pose.npy'), np.stack(T_T_W, 0))