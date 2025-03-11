import os
import queue
import time

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import pyzed.sl as sl

# Initialize the ZED camera
zed = sl.Camera()
init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, depth_mode=sl.DEPTH_MODE.ULTRA)
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ Failed to open ZED camera")
    exit(1)

runtime_params = sl.RuntimeParameters()
image = sl.Mat()
depth = sl.Mat()
point_cloud = sl.Mat()
imu_data = sl.SensorsData()

# Create output directory
output_dir = "ZED2i_Data"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/PointCloud", exist_ok=True)

# Video Writers
frame_width, frame_height = 1280, 720
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_writer_rgb = cv2.VideoWriter(f"{output_dir}/LeftVideo.mp4", fourcc, fps, (frame_width, frame_height))
video_writer_depth = cv2.VideoWriter(f"{output_dir}/DepthVideo.mp4", fourcc, fps, (frame_width, frame_height))

# IMU Data Storage
imu_list = []
point_cloud_queue = queue.Queue()

start_time = time.time()
frame_count = 0

while time.time() - start_time < 50:  # Run for 10 seconds
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        frame_count += 1

        # Capture RGB Image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        rgb_image = image.get_data()
        rgb_frame = cv2.cvtColor(rgb_image[:, :, :3], cv2.COLOR_RGB2BGR)
        video_writer_rgb.write(rgb_frame)

        # Capture Depth Map
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        depth_map = depth.get_data()
        depth_display = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=255 / 5), cv2.COLORMAP_JET)
        video_writer_depth.write(depth_display)

        # Capture Point Cloud (saved after recording)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        point_cloud_data = point_cloud.get_data()
        xyz = point_cloud_data[:, :, :3].reshape(-1, 3)
        colors = rgb_image[:, :, :3].reshape(-1, 3) / 255.0  # Normalize RGB

        valid_mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid_mask]
        colors = colors[valid_mask]

        if xyz.shape[0] > 0:
            point_cloud_queue.put((xyz, colors, frame_count))

        # Capture IMU Data (saved after recording)
        zed.get_sensors_data(imu_data, sl.TIME_REFERENCE.CURRENT)
        imu_values = imu_data.get_imu_data()
        acceleration = imu_values.get_linear_acceleration()
        angular_velocity = imu_values.get_angular_velocity()
        imu_list.append({
            "timestamp": time.time(),
            "acceleration_x": acceleration[0],
            "acceleration_y": acceleration[1],
            "acceleration_z": acceleration[2],
            "angular_velocity_x": angular_velocity[0],
            "angular_velocity_y": angular_velocity[1],
            "angular_velocity_z": angular_velocity[2]
        })

        # Display RGB and Depth Maps
        cv2.imshow("RGB Image", rgb_frame)
        cv2.imshow("Depth Map", depth_display)

        print(f"✅ Frame {frame_count} collected.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
# Save IMU Data and Point Cloud after recording
imu_df = pd.DataFrame(imu_list)
imu_df.to_csv(f"{output_dir}/imu_data.csv", index=False)

while not point_cloud_queue.empty():
    xyz, colors, frame_idx = point_cloud_queue.get()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(f"{output_dir}/PointCloud/frame_{frame_idx}.ply", pcd)

# Release Resources
video_writer_rgb.release()
video_writer_depth.release()
zed.close()
cv2.destroyAllWindows()

print("\n✅ Data collection completed!")
print(f"📂 Left Video saved as '{output_dir}/LeftVideo.mp4'")
print(f"📂 Depth Video saved as '{output_dir}/DepthVideo.mp4'")
print(f"📂 Point Clouds saved in '{output_dir}/PointCloud/' folder")
print(f"📂 IMU Data saved as '{output_dir}/imu_data.csv'")
