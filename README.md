<div align="center">

![Project Banner](./assets/project_banner.png)

# 🌌 ZED2i Perception & Data Intelligence
**Advanced Stereo Vision Pipeline & Synchronized Multi-Camera Acquisition**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![ZED SDK](https://img.shields.io/badge/ZED_SDK-4.0+-green.svg?style=for-the-badge)](https://www.stereolabs.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

---
</div>

## 🔭 Project Overview

This repository serves as a professional-grade ecosystem for **StereoLabs ZED2i** and **Intel RealSense D435i** integration. It bridges the gap between raw spatial data acquisition and high-level AI perception.

### 🧩 Core Ecosystem

```mermaid
graph TD
    Root[ZED2i Project Root] --> Stereo[Stereo Depth Intelligence]
    Root --> Data[Data Collection Suite]
    
    Stereo --> MaskRCNN[Mask R-CNN Detection]
    Stereo --> Hungarian[Hungarian Matching]
    Stereo --> Distance[CM-Level Depth]
    
    Data --> Dual[Synchronized ZED/RealSense]
    Data --> IMU[High-Freq IMU Logging]
    Data --> Specialized[Custom Collection Options]
```

---

## 📂 Navigation & Modules

| Module | Description | Key Documentation |
| :--- | :--- | :--- |
| **Perception** | Deep learning based depth estimation using stereo pairs. | [🧠 Perception README](stereoImageProcessing/code/StereoCameraFindingDepth/README.md) |
| **Data Engine** | Synchronized multi-camera recording (ZED + RealSense). | [📹 Data Engine README](default/code/Working/README.md) |
| **Specialized** | Custom acquisition profiles for specific research. | [🛠️ Specialized README](default/code/Working/samsang/collectionData/README.md) |

---

## 🚀 Quick Start

> [!TIP]
> Ensure you have the ZED SDK and RealSense SDK installed on your system before running the collection scripts.

### 1. Synchronized Data Collection
```bash
# Capture from ZED and RealSense simultaneously
python default/code/Working/script.py --fps 30 --output_dir ./recordings
```

### 2. Stereo Depth Finding
```bash
# Analyze stereo images for object distances
python stereoImageProcessing/code/StereoCameraFindingDepth/Stereo_Image_All2.py
```

---

## 🛠️ Technology Stack

- **Computer Vision**: OpenCV, StereoLabs SDK (pyzed), Intel RealSense SDK (pyrealsense2)
- **Deep Learning**: PyTorch, Torchvision (Mask R-CNN ResNet50)
- **Data Science**: NumPy, Pandas, SciPy (Linear Sum Assignment)
- **Visualization**: Matplotlib

---
<div align="center">
Designed for precision spatial intelligence. 
</div>
