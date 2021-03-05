# 3D multimodal medical imaging technique based on freehand ultrasound and structured light

This is the main repository for the 3D multimodal imaging technique using freehand ultrasound and 3D reconstruction by strucutred light. We the proposed technique can obtain the internal strcuture with 3D freehand ultrasound and complement this information the interanl features acquired with strcutured light. Our system is composed of two cameras, a projector, and a ultrasound machine as shown the following figure.

<p align="center">
    <img src="figures/system.png" alt="multimodal-system" width="500px"/>
</p>

The 3D freehand ultrasound system is composed of the stereo vision system and the ultrasound machine. The stereo-vision system {Cam1}-{cam2} and a target of three coplanar circles attached in the probe are used for pose estimation for freehand ultrasound reconstruction. Point detection of the target is addressed with a convolutional neural network model. {Cam1} and the projector {P} are used for external 3D reconstruction using strucutred light techniques.

The following codes are available:
* [3D freehand ultrasound calibration](https://github.com/jhacsonmeza/US-Calibration)
* [Stereo vision and ultrasound simultenous image acquisition](https://github.com/jhacsonmeza/StereoBaslerUltrasound)