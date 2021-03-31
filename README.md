# 3D multimodal medical imaging technique based on freehand ultrasound and structured light

This is the main repository for the 3D multimodal imaging technique using freehand ultrasound and 3D reconstruction by structured light. With the proposed technique, we can obtain the internal structure with 3D freehand ultrasound and complement this information with the external features acquired with structured light. Our system is composed of two cameras, a projector, and an ultrasound machine as shown in the following figure.

<p align="center">
    <img src="figures/system.png" alt="multimodal-system" width="500px"/>
</p>

The 3D freehand ultrasound system is composed of the stereo vision system and the ultrasound machine. The stereo vision system {Cam1} - {cam2} and a target of three coplanar circles attached in the probe are used for pose estimation for freehand ultrasound reconstruction. Point detection of the target is addressed with a deep neural network model. {Cam1} and the projector {P} are used for external 3D reconstruction using structured light techniques. The acquired information with both modalities is referred to the same world frame, which is {Cam1}. In this way, we avoid data registration.

In this repo, code for 3D multimodal reconstruction is available. The following codes are also available:
* [3D freehand ultrasound calibration](https://github.com/jhacsonmeza/US-Calibration)
* [Stereo vision and ultrasound simultenous image acquisition](https://github.com/jhacsonmeza/StereoBaslerUltrasound)

## Dependencies

* Python 3
* NumPy
* OpenCV
* PyTorch
* Numba

## Structure of the repository
`sl_us/` folder contains the modules needed for structured light (`sl3d.py`) and freehand ultrasound (`us3d.py`) reconstruction, phase recovery (`centerline.py`), and some utility functions (`utils.py`). Furthermore, `marker/` folder contains `models.py` where deep neural networks for pose estimation are defined and `utils.py` where some functions are defined to aid pose estimation. Finally, `ProbePose.py` and `multimodal_recons.py` are the scripts for pose estimation of the marker for freehand ultrasound and multimodal 3D reconstruction, respectively.

## Usage example

With this example, you can recreate the multimodal reconstruction results shown in the paper. For that, we use a breast phantom [3B SONOtrain P125](https://www.3bscientific.com/sonotrain-breast-model-with-tumours-1019635-p125-3b-scientific,p_1397_27469.html), shown in the following figure.

<p align="center">
    <img src="figures/breast_phantom.png" alt="breast-phantom" width="500px"/>
</p>

This model contains three breast tumors in the positions shown in the image. 170x130x55 mm is the phantom's dimensions. For this example, the external surface of the breast is reconstructed with structured light patterns. The internal three tumors are also reconstructed with the 3D freehand ultrasound method, for that, different images of each tumor were acquired with slight displacements between frames.

To run this example:

* Clone this repository: `git clone https://github.com/jhacsonmeza/StructuredLight_3DfreehandUS.git`
* `cd StructuredLight_3DfreehandUS`
* You need to download the `dataset/` folder of projected patterns and freehand ultrasound acquisition of the phantom [here](https://drive.google.com/drive/folders/1abZWgODq2XpUyIZclIMR59ldTdKFfqie?usp=sharing). In this folder you can find `SLdata/` with the images for structured light reconstruction, `USdata` with images for pose estimation of the probe, and ultrasound B-scans for 3D reconstruction. Furthermore, you can find camera-projector stereo parameters (`cam1_proj_params.npz`), camera1-camera2 stereo parameters (`cam1_cam2_params.npz`) for pose estimation, and freehand ultrasound calibration parameters (`US_params.npz`). Finally, the weights of the deep neural network models `superpoint.pt` and `ellipsegnet.pt` for pose estimation.  
* Run `python ProbePose.py` for pose estimation of the marker/ultrasound transducer. This snippet of code shows the estimated pose for each ultrasound scan acquired. It generates a file `target_pose.npy` with the position and orientation of the marker relative to {Cam1}.
* As the last stage, run `python multimodal_recons.py` for multimodal 3D reconstruction of the external surface of the breast phantom and the internal three tumors. It will generate a file `SL_US_3D.ply`, which you can visualize, for example, with the open-source software [MeshLab](https://www.meshlab.net/).

A visualization example of the generated point cloud is shown below.

<p align="center">
    <img src="figures/phantom_SL+US.png" alt="phantom-SL-US.png" width="800px"/>
</p>