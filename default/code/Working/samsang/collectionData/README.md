# Specialized Data Collection

This subdirectory contains specialized tools for ZED camera data acquisition, likely used for specific dataset gathering (e.g., "samsang" project).

## Key Component

### `CollectData.py`
A highly configurable script for capturing ZED camera data with flexible saving options.

#### Features:
- **Save Options**: Can be configured to save different subsets of data:
  - `All`: Saves Color Videos (Left/Right), Depth Video, and IMU data.
  - `ColorImages`: Saves only left/right color videos.
  - `IMU`: Saves only the `imu_data.csv`.
  - `RGBDepth`: Saves color and depth videos.
- **Performance**: Uses optimized OpenCV settings and multi-threading for IMU logging to minimize frame drops.
- **Positional Tracking**: Enables ZED's positional tracking to log camera translation (X, Y, Z) alongside IMU acceleration and angular velocity.

## Usage
```bash
python CollectData.py --output_dir ./ProjectData --save_option All --fps 30
```

### Arguments:
- `--output_dir`: Where to save the captured files.
- `--resolution`: Camera resolution (`HD720`, `HD1080`, `HD2K`).
- `--fps`: Supported values: `15`, `30`, `60`, `120`.
- `--save_option`: `All`, `ColorImages`, `IMU`, or `RGBDepth`.
