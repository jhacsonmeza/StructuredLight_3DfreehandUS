import os
import cv2
import torch
import glob
import numpy as np

from sl_us import utils
import marker


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create SuperPoint model
detect = marker.SuperPointNet(3)
detect.load_state_dict(torch.load('dataset/superpoint.pt',map_location=device))

# Create EllipSegNet model
ellipseg = marker.EllipSegNet(16, 1)
ellipseg.load_state_dict(torch.load('dataset/ellipsegnet.pt',map_location=device))

# Create Detector model
centerDetect = marker.Detector(detect, ellipseg, (320,240), 120)
centerDetect.to(device)
centerDetect.eval()




# Root path
base = os.path.relpath('dataset/USdata')


# Set window name and size
cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 1792, 717)

# Read left and right images of the target
I1 = utils.natsort(glob.glob(os.path.join(base,'L','*')))
I2 = utils.natsort(glob.glob(os.path.join(base,'R','*')))

# Load stereo calibration parameters
Params = np.load(os.path.join(os.path.dirname(base),'cam1_cam2_params.npz'))
K1 = Params['K1']
K2 = Params['K2']
R = Params['R']
t = Params['t']
F = Params['F']
dist1 = Params['dist1']
dist2 = Params['dist2']

# Create projection matrices of camera 1 and camera 2
P1 = K1 @ np.c_[np.eye(3), np.zeros(3)]
P2 = K2 @ np.c_[R, t]


T_T_W = []
for im1n, im2n in zip(I1,I2):
    im1 = cv2.imread(im1n)
    im2 = cv2.imread(im2n)
    
    # Target detection
    c1 = centerDetect(cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY))
    c2 = centerDetect(cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY))
    
    # Undistort 2D center coordinates in each image
    c1 = cv2.undistortPoints(c1.reshape(-1,1,2), K1, dist1, None, None, K1)
    c2 = cv2.undistortPoints(c2.reshape(-1,1,2), K2, dist2, None, None, K2)
    
    # Estimate 3D coordinate of the concentric circles through triangulation
    X = cv2.triangulatePoints(P1, P2, c1, c2)
    X = X[:3]/X[-1] # Convert coordinates from homogeneous to Euclidean

    # Target pose estimation relative to the left camera/world frame
    Xo, Xx, Xy = X.T
    R_T_W, t_T_W = marker.getPose(Xo, Xx, Xy)

    
    # Save pose
    T_T_W.append(np.r_[np.c_[R_T_W, t_T_W], [[0,0,0,1]]])
    
    
    ############################ Visualize results ############################
    marker.drawAxes(im1, K1, dist1, R_T_W, t_T_W)
    
    cv2.imshow('Detection',np.hstack([im1,im2]))
    if cv2.waitKey(255) == 27:
        break

cv2.destroyAllWindows()


# Save variables
np.save(os.path.join(base,'target_pose.npy'), np.stack(T_T_W, 0))