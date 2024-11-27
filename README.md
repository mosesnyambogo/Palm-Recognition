<div align="center">
<img alt="" src="https://github.com/Faceplugin-ltd/FaceRecognition-Javascript/assets/160750757/657130a9-50f2-486d-b6d5-b78bcec5e6e2.png" width=200/>
</div>

# Palm Recognition SDK for Windows and Linux - Fully On Premise

## Overview
The world's 1st **Completely Free and Open Source** `Palm Recognition SDK` from [Faceplugin](https://faceplugin.com/) for developers to integrate palm recognition capabilities into applications. Supports real-time, high-accuracy palm recognition with deep learning models.
<br>This is `on-premise palm recognition SDK` which means everything is processed in your device and **NO** data leaves it.
<br>You can use this SDK on Windows and Linux.
<br><br>**Please contact us if you need the SDK with higher accuracy.**
<br></br>

## Key Features
- **Real-Time Palm Recognition**: Can detect and recognize palm from live video streams. Currently only supports palm recognition from an image.
- **High Accuracy**: Built with deep learning models trained on large datasets.
- **Cross-Platform**: Compatible with Windows and Linux.
- **Flexible Integration**: Easy-to-use APIs for seamless integration into any project.
- **Scalable**: Works on local devices, cloud, or embedded systems.
- **Python SDK**: Comprehensive support for Python with extensive documentation and examples.

## Applications
This **Palm Recognition SDK** is ideal for a wide range of applications, including:
- **Time Attendance Systems**: Monitor arrivals and depatures using palm recognition.
- **Security Systems**: Access control and surveillance.
- **User Authentication**: Biometric login and multi-factor authentication.
- **Smart Devices**: Integration into IoT devices for smart home or office applications.
- **Augmented Reality**: Enhance AR applications with real-time palm recognition.

## Installation
Please download anaconda on your computer and install it.
We used Windows machine without GPU for testing.

### create anaconda environment 
conda create -n palm python=3.9

### activate env
conda activate palm

### install dependencies
pip install torch torchvision torchaudio
pip install opencv-python
pip install tqdm
pip install scikit-image
pip install mediapipe

### compare two palm images in the test_images directory.
python main.py

## APIs and Parameters

**classify_hand(mp_hands, hand_landmarks, image_width):** determine if the hand is left hand or right hand<br>
**extract_roi(hands, mp_hands, img_path):** extract region of interest from the palm image for template matching<br>
**extract_features(mp_hands, hands, path: str):**: extract template from the plam image specified by the path parameter.
**compare_two_images(mp_hands, hands, image_path1, image_path2, similarity_threshold=0.8)**: compare two hand images to determine if they are the same hand or not.

## List of our Products

* **[FaceRecognition-LivenessDetection-Android](https://github.com/Faceplugin-ltd/FaceRecognition-Android)**
* **[FaceRecognition-LivenessDetection-iOS](https://github.com/Faceplugin-ltd/FaceRecognition-iOS)**
* **[FaceRecognition-LivenessDetection-Javascript](https://github.com/Faceplugin-ltd/FaceRecognition-LivenessDetection-Javascript)**
* **[FaceLivenessDetection-Android](https://github.com/Faceplugin-ltd/FaceLivenessDetection-Android)**
* **[FaceLivenessDetection-iOS](https://github.com/Faceplugin-ltd/FaceLivenessDetection-iOS)**
* **[FaceLivenessDetection-Linux](https://github.com/Faceplugin-ltd/FaceLivenessDetection-Linux)**
* **[FaceRecognition-LivenessDetection-React](https://github.com/Faceplugin-ltd/FaceRecognition-LivenessDetection-React)**
* **[FaceRecognition-LivenessDetection-Vue](https://github.com/Faceplugin-ltd/FaceRecognition-LivenessDetection-Vue)**
* **[Face Recognition SDK](https://github.com/Faceplugin-ltd/Face-Recognition-SDK)**
* **[Liveness Detection SDK](https://github.com/Faceplugin-ltd/Face-Liveness-Detection-SDK)**
* **[ID Card Recognition](https://github.com/Faceplugin-ltd/ID-Card-Recognition)**

## Contact
<div align="left">
<a target="_blank" href="mailto:info@faceplugin.com"><img src="https://img.shields.io/badge/email-info@faceplugin.com-blue.svg?logo=gmail " alt="faceplugin.com"></a>&emsp;
<a target="_blank" href="https://t.me/faceplugin"><img src="https://img.shields.io/badge/telegram-@faceplugin-blue.svg?logo=telegram " alt="faceplugin.com"></a>&emsp;
<a target="_blank" href="https://wa.me/+14422295661"><img src="https://img.shields.io/badge/whatsapp-faceplugin-blue.svg?logo=whatsapp " alt="faceplugin.com"></a>
</div>

