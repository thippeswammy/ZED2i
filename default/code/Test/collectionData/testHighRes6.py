import os
import queue
import threading
import time

import cv2
import numpy as np
import pandas as pd
import pyzed.sl as sl

# Initialize the ZED camera
zed = sl.Camera()
init_params = sl.InitParameters(
    camera_resolution=sl.RESOLUTION.HD720,
    depth_mode=sl.DEPTH_MODE.ULTRA,
    coordinate_units=sl.UNIT.METER  # Ensure position is in meters
)

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ Failed to open ZED camera")
    exit(1)

# Enable Positional Tracking
tracking_params = sl.PositionalTrackingParameters()
zed.enable_positional_tracking(tracking_params)

runtime_params = sl.RuntimeParameters()
image_left = sl.Mat()
image_right = sl.Mat()
depth = sl.Mat()
point_cloud = sl.Mat()
imu_data = sl.SensorsData()
pose = sl.Pose()

# Create output directories
output_dir = "./Data/ZED2i_Data"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/PointCloud", exist_ok=True)

# Video Writers
frame_width, frame_height = 1280, 720
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_writer_left = cv2.VideoWriter(f"{output_dir}/LeftVideo.mp4", fourcc, fps, (frame_width, frame_height))
video_writer_right = cv2.VideoWriter(f"{output_dir}/RightVideo.mp4", fourcc, fps, (frame_width, frame_height))
video_writer_depth = cv2.VideoWriter(f"{output_dir}/DepthVideo.mp4", fourcc, fps, (frame_width, frame_height))

# Queues for saving tasks
imu_queue = queue.Queue()
pcd_queue = queue.Queue()

# IMU Data Storage
imu_list = []


# Function to save IMU Data asynchronously
def save_imu_data():
    while True:
        data = imu_queue.get()
        if data is None:
            break
        imu_list.append(data)


# Start saving threads
imu_thread = threading.Thread(target=save_imu_data, daemon=True)
imu_thread.start()

# FPS Tracking
start_time = time.time()
frame_count = 0
start_fps_time = time.time()
frame_count_per_sec = 0

while cv2.waitKey(1) != ord('q'):  # Run until 'q' is pressed
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        frame_count += 1
        frame_count_per_sec += 1

        # Capture Left Image
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        left_image = image_left.get_data()
        left_frame = cv2.cvtColor(left_image[:, :, :3], cv2.COLOR_RGB2BGR)
        video_writer_left.write(left_frame)

        # Capture Right Image
        zed.retrieve_image(image_right, sl.VIEW.RIGHT)
        right_image = image_right.get_data()
        right_frame = cv2.cvtColor(right_image[:, :, :3], cv2.COLOR_RGB2BGR)
        video_writer_right.write(right_frame)

        # Capture Depth Map
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        depth_map = depth.get_data()
        depth_map[np.isnan(depth_map)] = 0  # Handle NaN values
        depth_display = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=255 / 5), cv2.COLORMAP_JET)
        video_writer_depth.write(depth_display)

        # Capture IMU Data
        zed.get_sensors_data(imu_data, sl.TIME_REFERENCE.CURRENT)
        imu_values = imu_data.get_imu_data()
        acceleration = imu_values.get_linear_acceleration()
        angular_velocity = imu_values.get_angular_velocity()

        # Capture Position Data (X, Y, Z)
        zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        translation = pose.get_translation()
        position_x, position_y, position_z = translation.get()

        imu_queue.put({
            "timestamp": time.time(),
            "position_x": position_x,
            "position_y": position_y,
            "position_z": position_z,
            "acceleration_x": acceleration[0],
            "acceleration_y": acceleration[1],
            "acceleration_z": acceleration[2],
            "angular_velocity_x": angular_velocity[0],
            "angular_velocity_y": angular_velocity[1],
            "angular_velocity_z": angular_velocity[2]
        })

        # # Display FPS every second
        # if time.time() - start_fps_time >= 1.0:  # Every second
        #     print(f"📷 Frames saved in last second: {frame_count_per_sec}")
        #     frame_count_per_sec = 0
        #     start_fps_time = time.time()

        # Display RGB and Depth Maps
        cv2.imshow("Left Image", left_frame)
        # cv2.imshow("Right Image", right_frame)
        # cv2.imshow("Depth Map", depth_display)

print("Wait...")

total_time = time.time() - start_time
fps_avg = frame_count / total_time
# Stop threads
imu_queue.put(None)

video_writer_left.release()
video_writer_right.release()
video_writer_depth.release()

# Release Resources
print("Video saved. Wait...")

# Save IMU Data after all frames
imu_df = pd.DataFrame(imu_list)
imu_df.to_csv(f"{output_dir}/imu_data.csv", index=False)
print("IMU data saved. Wait...")

imu_thread.join()
print("IMU thread stopped. Wait...")

zed.close()
cv2.destroyAllWindows()

# Calculate total FPS
print("\n✅ Data collection completed!")
print(f"📂 Left Video saved as '{output_dir}/LeftVideo.mp4'")
print(f"📂 Right Video saved as '{output_dir}/RightVideo.mp4'")
print(f"📂 Depth Video saved as '{output_dir}/DepthVideo.mp4'")
print(f"⏱️ Total Time: {total_time:.2f} sec")
print(f"📷 Average FPS: {fps_avg:.2f} frames/sec")
