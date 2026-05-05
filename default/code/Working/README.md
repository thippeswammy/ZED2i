# Camera Data Collection (ZED & RealSense)

This directory contains scripts for synchronized data collection from ZED2i and Intel RealSense D435i cameras.

## Main Scripts

### 1. Dual Camera Recorder
- **`script.py`**: A robust data collection tool that captures data from **both** ZED2i and RealSense D435i simultaneously.
  - **Video Streams**: Records Color and Depth videos (MP4) for both cameras.
  - **IMU Data**: Logs positional data, acceleration, and angular velocity to `imu_data.csv`.
  - **Synchronization**: Uses threading and events to ensure cameras are initialized and recording in parallel.
  - **Output**: Automatically creates unique versioned folders (e.g., `Data2/ZED2i_Data_1`, `Data2/RealSense_D435i_1`).

### 2. Camera-Specific Variations
- **`ZedCamera/`**: Contains scripts dedicated to ZED camera operations.
  - `script.py`, `script2.py`, etc., represent different iterations of ZED data collection and processing.
- **`ZedIntelReal/`**: Additional scripts for combined ZED and Intel RealSense processing.

## Command Line Arguments (`script.py`)
- `--output_dir`: Base directory for saved data (default: `./Data2`).
- `--resolution`: ZED resolution (`HD720`, `HD1080`, `HD2K`).
- `--fps`: Frames per second (default: `30`).
- `--depth_mode`: ZED depth quality (`PERFORMANCE`, `ULTRA`).

## Usage
Run the main script to start recording from both cameras:
```bash
python script.py --output_dir ./MyData --fps 30
```
Press **'q'** in any display window to stop recording.

## Requirements
- `pyzed` (ZED SDK)
- `pyrealsense2` (Intel RealSense SDK)
- `opencv-python`
- `pandas`
- `numpy`
