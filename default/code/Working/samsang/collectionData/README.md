# 🛠️ Specialized Collection Suite

Custom-tailored acquisition profiles for the **ZED2i** camera, optimized for specific dataset requirements and research projects (e.g., Samsung Perception Project).

---

## 🚀 Configurable Capture: `CollectData.py`

This script provides granular control over which data streams are prioritized, allowing for reduced CPU overhead when specific sensors aren't required.

### 📡 Available Profiles

> [!NOTE]
> All profiles utilize multi-threaded IMU writing to ensure zero frame-drop during high-speed movement.

| Profile | Description | Saved Artifacts |
| :--- | :--- | :--- |
| `All` | Complete spatial dataset. | Color (L/R), Depth Video, IMU CSV |
| `ColorImages` | High-fidelity visual only. | Left & Right MP4 |
| `IMU` | Motion-only capture. | imu_data.csv |
| `RGBDepth` | Visual and depth fusion. | Left Color, Depth Map MP4 |

---

## 🛠️ Optimization Features

- **⚡ Optimized OpenCV**: Explicitly uses `cv2.setUseOptimized(True)` and sets thread counts for efficient image processing.
- **🎯 Positional Tracking**: Enables ZED's spatial mapping to provide absolute world-coordinates ($X, Y, Z$) in addition to raw IMU readings.
- **⚖️ Quality Control**: Enforces confidence thresholds (`texture_confidence_threshold`) to ensure depth maps only contain high-reliability data.

---

## 📖 Execution Guide

```bash
# Example: High-speed IMU and Color capture
python CollectData.py --output_dir ./Session_01 --save_option All --fps 60 --resolution HD720
```

### Argument Reference
- `--resolution`: Standard ZED formats (`HD720`, `HD1080`, `HD2K`).
- `--fps`: Valid choices: `15`, `30`, `60`, `120`.
- `--save_option`: Selection from the profiles listed above.

---
> [!WARNING]
> High FPS modes (60/120) require significant disk I/O throughput. Using an SSD for the `--output_dir` is highly recommended.
