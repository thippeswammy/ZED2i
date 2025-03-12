import os
import queue
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl

# Enable OpenCV optimizations
cv2.setUseOptimized(True)
cv2.setNumThreads(8)

# Initialize the ZED camera
zed = sl.Camera()
init_params = sl.InitParameters(
    camera_resolution=sl.RESOLUTION.HD1080,  # High resolution (1920x1080)
    depth_mode=sl.DEPTH_MODE.PERFORMANCE,  # Fastest depth processing
    coordinate_units=sl.UNIT.METER,
    camera_fps=30  # Higher FPS for smoothness
)

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit(1)

# Enable Positional Tracking
tracking_params = sl.PositionalTrackingParameters()
zed.enable_positional_tracking(tracking_params)

# Set Runtime Parameters
runtime_params = sl.RuntimeParameters()
runtime_params.enable_depth = True  # Use only if needed
runtime_params.confidence_threshold = 80  # Improves speed
runtime_params.texture_confidence_threshold = 80

# Allocate Matrices
image_left = sl.Mat()
image_right = sl.Mat()
depth = sl.Mat()
imu_data = sl.SensorsData()
pose = sl.Pose()

# Create output directories
output_dir = "./Data/ZED2i_Data"
os.makedirs(output_dir, exist_ok=True)

# Video Writers
frame_width, frame_height = 1280, 720  # VGA resolution
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_writer_left = cv2.VideoWriter(f"{output_dir}/LeftVideo.mp4", fourcc, fps, (frame_width, frame_height))
video_writer_right = cv2.VideoWriter(f"{output_dir}/RightVideo.mp4", fourcc, fps, (frame_width, frame_height))
video_writer_depth = cv2.VideoWriter(f"{output_dir}/DepthVideo.mp4", fourcc, fps, (frame_width, frame_height))

# Queues for saving tasks
imu_queue = queue.Queue()

# IMU Data Storage
imu_file = open(f"{output_dir}/imu_data.csv", "w")
imu_file.write(
    "timestamp,position_x,position_y,position_z,acceleration_x,acceleration_y,acceleration_z,angular_velocity_x,angular_velocity_y,angular_velocity_z\n")


# Function to save IMU Data asynchronously
def save_imu_data():
    while True:
        data = imu_queue.get()
        if data is None:
            break
        imu_file.write(f"{data['timestamp']},{data['position_x']},{data['position_y']},{data['position_z']},"
                       f"{data['acceleration_x']},{data['acceleration_y']},{data['acceleration_z']},"
                       f"{data['angular_velocity_x']},{data['angular_velocity_y']},{data['angular_velocity_z']}\n")


# Start saving threads
imu_thread = threading.Thread(target=save_imu_data, daemon=True)
imu_thread.start()


# Function to write video asynchronously
def write_video(writer, frame):
    writer.write(frame)


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

        # Capture Right Image
        zed.retrieve_image(image_right, sl.VIEW.RIGHT)
        right_image = image_right.get_data()
        right_frame = cv2.cvtColor(right_image[:, :, :3], cv2.COLOR_RGB2BGR)

        # Capture Depth Map
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        depth_map = depth.get_data()
        depth_map[np.isnan(depth_map)] = 0  # Handle NaN values
        depth_display = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=255 / 5), cv2.COLORMAP_JET)

        # Save video frames in separate threads
        threading.Thread(target=write_video, args=(video_writer_left, left_frame)).start()
        threading.Thread(target=write_video, args=(video_writer_right, right_frame)).start()
        threading.Thread(target=write_video, args=(video_writer_depth, depth_display)).start()

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

        # Display RGB and Depth Maps
        cv2.imshow("Left Image", left_frame)
        cv2.imshow("Right Image", right_frame)
        cv2.imshow("Depth Map", depth_display)

print("Wait...")

total_time = time.time() - start_time
fps_avg = frame_count / total_time

# Stop threads
imu_queue.put(None)
imu_thread.join()
imu_file.close()

video_writer_left.release()
video_writer_right.release()
video_writer_depth.release()

# Release Resources
print("Video saved. Wait...")

print("\n✅ Data collection completed!")
print(f"📂 Left Video saved as '{output_dir}/LeftVideo.mp4'")
print(f"📂 Right Video saved as '{output_dir}/RightVideo.mp4'")
print(f"📂 Depth Video saved as '{output_dir}/DepthVideo.mp4'")
print(f"📂 IMU Data saved as '{output_dir}/imu_data.csv'")
print(f"⏱️ Total Time: {total_time:.2f} sec")
print(f"📷 Average FPS: {fps_avg:.2f} frames/sec")

zed.close()
cv2.destroyAllWindows()
