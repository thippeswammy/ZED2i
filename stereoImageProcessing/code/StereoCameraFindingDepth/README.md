# Stereo Camera Depth Finding

This directory contains Python scripts and Jupyter notebooks for performing object detection and depth estimation using stereo image pairs captured from a ZED camera (or similar stereo setups).

## Key Components

### 1. Object Detection & Depth Estimation
- **`stereoimagedepthfinding.py`**: A comprehensive script that uses a Mask R-CNN model (ResNet-50-FPN V2) for object detection and instance segmentation. 
  - **Functionality**: It matches objects between left and right images using a cost function based on vertical/horizontal displacement and area differences.
  - **Matching**: Uses the Hungarian algorithm (`linear_sum_assignment`) for optimal object pairing.
  - **Disparity**: Calculates horizontal disparity to estimate the distance of objects from the camera.
- **`Stereo_Image_All2.py`**: An optimized version of the depth-finding logic.
  - Includes specific calibration constants (e.g., `FocalLength`, `tanTheta`) to output real-world depth measurements in centimeters (cm).
  - Uses a pre-trained `MaskrCNN_model.pt` if available.
- **`Stereo_Image_v_282.ipynb` & `Stereo_Image_v_167.ipynb`**: Notebook versions of the implementation, useful for interactive experimentation and visualization.

### 2. Image Capture
- **`ZedLeftRightImageCapture.py`**: A utility script to capture and save synchronized left and right image frames from a ZED camera.
  - **Controls**: Press **'s'** to save a pair of images, **'q'** to quit.
  - Saves images as `left_image_0X.jpg` and `right_image_0X.jpg`.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- PyTorch & Torchvision
- NumPy, Matplotlib, SciPy
- ZED SDK (`pyzed`)

## How it Works
1. **Capture**: Stereo images are captured (or loaded).
2. **Detection**: Mask R-CNN identifies objects in both images.
3. **Tracking/Matching**: The system calculates a "cost" for pairing any object in the left image with any object in the right image.
4. **Distance Calculation**: Based on the horizontal shift (disparity) between the matched objects and camera calibration parameters, the distance to each object is computed.
