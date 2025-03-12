import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl


def parse_args():
    parser = argparse.ArgumentParser(description="ZED Camera Data Collection")
    parser.add_argument("--output_dir", type=str, default="./Data/ZED2i_Data", help="Output directory")
    parser.add_argument("--resolution", type=str, default="HD1080", choices=["HD720", "HD1080", "HD2K"],
                        help="Camera resolution")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (Must be supported by ZED)")
    parser.add_argument("--depth_mode", type=str, default="PERFORMANCE", choices=["PERFORMANCE", "ULTRA"],
                        help="Depth mode")
    parser.add_argument("--save_option", type=str, default="All", choices=["All", "ColorImages", "IMU", "RGBDepth"],
                        help="Data to save")

    args = parser.parse_args()

    if args.fps not in [15, 30, 60, 120]:  # Validate FPS (adjust based on ZED model)
        parser.error("--fps must be one of [15, 30, 60, 120]")

    return args


def initialize_camera(args):
    zed = sl.Camera()
    resolution_dict = {
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD2K": sl.RESOLUTION.HD2K
    }
    depth_mode_dict = {
        "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
        "ULTRA": sl.DEPTH_MODE.ULTRA
    }
    init_params = sl.InitParameters(
        camera_resolution=resolution_dict[args.resolution],
        depth_mode=depth_mode_dict[args.depth_mode],
        coordinate_units=sl.UNIT.METER,
        camera_fps=args.fps
    )

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Failed to open ZED camera")

    return zed


def create_output_directories(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def initialize_video_writers(args, output_dir):
    frame_width, frame_height = 1280, 720  # Adjust based on resolution if needed
    fps = args.fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writers = {}
    if args.save_option in ["All", "ColorImages", "RGBDepth"]:
        video_writers["left"] = cv2.VideoWriter(f"{output_dir}/LeftVideo.mp4", fourcc, fps, (frame_width, frame_height))
        video_writers["right"] = cv2.VideoWriter(f"{output_dir}/RightVideo.mp4", fourcc, fps,
                                                 (frame_width, frame_height))
        video_writers["depth"] = cv2.VideoWriter(f"{output_dir}/DepthVideo.mp4", fourcc, fps,
                                                 (frame_width, frame_height))

    return video_writers


def initialize_imu_file(output_dir):
    imu_file = open(f"{output_dir}/imu_data.csv", "w")
    imu_file.write("timestamp,position_x,position_y,position_z,"
                   "acceleration_x,acceleration_y,acceleration_z,"
                   "angular_velocity_x,angular_velocity_y,angular_velocity_z\n")
    return imu_file


def save_imu_data(imu_file, imu_queue):
    while True:
        data = imu_queue.get()
        if data is None:
            break
        imu_file.write(f"{data['timestamp']},{data['position_x']},{data['position_y']},{data['position_z']},"
                       f"{data['acceleration_x']},{data['acceleration_y']},{data['acceleration_z']},"
                       f"{data['angular_velocity_x']},{data['angular_velocity_y']},{data['angular_velocity_z']}\n")
    imu_file.close()


def main():
    args = parse_args()
    cv2.setUseOptimized(True)
    cv2.setNumThreads(8)

    try:
        zed = initialize_camera(args)
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)

    runtime_params = sl.RuntimeParameters(enable_depth=True, confidence_threshold=80, texture_confidence_threshold=80)

    image_left, image_right, depth = sl.Mat(), sl.Mat(), sl.Mat()
    imu_data, pose = sl.SensorsData(), sl.Pose()

    output_dir = args.output_dir
    create_output_directories(output_dir)

    video_writers = initialize_video_writers(args, output_dir)

    imu_queue = queue.Queue()
    imu_file = initialize_imu_file(output_dir)

    imu_thread = threading.Thread(target=save_imu_data, args=(imu_file, imu_queue), daemon=True)
    imu_thread.start()

    start_time = time.time()
    frame_count = 0

    while cv2.waitKey(1) & 0xFF != ord('q'):
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            frame_count += 1

            if args.save_option in ["All", "ColorImages", "RGBDepth"]:
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                left_frame = cv2.cvtColor(image_left.get_data()[:, :, :3], cv2.COLOR_RGB2BGR)

                zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                right_frame = cv2.cvtColor(image_right.get_data()[:, :, :3], cv2.COLOR_RGB2BGR)

                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                depth_map = np.nan_to_num(depth.get_data(), nan=0)
                depth_display = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=255 / 5), cv2.COLORMAP_JET)

                video_writers["left"].write(left_frame)
                video_writers["right"].write(right_frame)
                video_writers["depth"].write(depth_display)

                cv2.imshow("Left Image", left_frame)
                cv2.imshow("Right Image", right_frame)
                cv2.imshow("Depth Map", depth_display)

            if args.save_option in ["All", "IMU"]:
                zed.get_sensors_data(imu_data, sl.TIME_REFERENCE.CURRENT)
                imu_values = imu_data.get_imu_data()
                acceleration, angular_velocity = imu_values.get_linear_acceleration(), imu_values.get_angular_velocity()

                zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                position_x, position_y, position_z = pose.get_translation().get()

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

    imu_queue.put(None)
    imu_thread.join()

    for writer in video_writers.values():
        writer.release()

    total_time = time.time() - start_time
    print(f"\n✅ Data collection completed in {total_time:.2f} sec with an average FPS of {frame_count / total_time:.2f}")

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
