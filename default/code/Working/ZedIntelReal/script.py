import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
import pyzed.sl as sl

start_time = time.time_ns()  # Initialize global start_time in nanoseconds


def get_unique_folder(base_dir, name):
    folder = os.path.join(base_dir, name)
    folder = os.path.normpath(folder)  # Normalize path for Windows
    count = 1
    while os.path.exists(folder):
        folder = os.path.normpath(os.path.join(base_dir, f"{name}_{count}"))
        count += 1
    os.makedirs(folder, exist_ok=True)
    return folder


class CameraRecorder:
    def __init__(self):
        self.running = False
        self.args = self.parse_args()
        self.zed_ready = threading.Event()
        self.realsense_ready = threading.Event()
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.realsense_data_dir = get_unique_folder(base_path,
                                                    os.path.join(self.args.output_dir, "RealSense_D435i"))
        self.zed_data_dir = get_unique_folder(base_path, os.path.join(self.args.output_dir, "ZED2i_Data"))
        print(self.zed_data_dir)
        print(self.realsense_data_dir)
        os.makedirs(self.zed_data_dir, exist_ok=True)
        self.zed_frame_queue = queue.Queue(maxsize=10)
        os.makedirs(self.realsense_data_dir, exist_ok=True)
        self.realsense_frame_queue = queue.Queue(maxsize=10)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Dual Camera Data Collection")
        parser.add_argument("--output_dir", type=str, default="./Data3", help="Base output directory")
        parser.add_argument("--resolution", type=str, default="HD720", choices=["HD720", "HD1080", "HD2K"],
                            help="ZED camera resolution")
        parser.add_argument("--fps", type=int, default=30, help="Frames per second")
        parser.add_argument("--depth_mode", type=str, default="ULTRA", choices=["PERFORMANCE", "ULTRA"],
                            help="ZED depth mode")
        return parser.parse_args()

    def initialize_realsense(self):
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, self.args.fps)
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, self.args.fps)
            config.enable_stream(rs.stream.gyro)
            config.enable_stream(rs.stream.accel)
            config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, self.args.fps)
            config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, self.args.fps)
            profile = pipeline.start(config)
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            self.realsense_ready.set()
            return pipeline, rs.align(rs.stream.color)
        except Exception as e:
            print(f"Failed to initialize RealSense: {str(e)}")
            self.realsense_ready.set()
            return None, None

    def initialize_zed(self):
        try:
            zed = sl.Camera()
            resolution_dict = {"HD720": sl.RESOLUTION.HD720, "HD1080": sl.RESOLUTION.HD1080, "HD2K": sl.RESOLUTION.HD2K}
            depth_mode_dict = {"PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE, "ULTRA": sl.DEPTH_MODE.ULTRA}
            init_params = sl.InitParameters(
                camera_resolution=resolution_dict[self.args.resolution],
                depth_mode=depth_mode_dict[self.args.depth_mode],
                coordinate_units=sl.UNIT.METER,
                camera_fps=self.args.fps
            )
            if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError("Failed to open ZED camera")
            zed.enable_positional_tracking(sl.PositionalTrackingParameters())
            if zed.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError("Failed to grab initial frame")
            self.zed_ready.set()
            return zed
        except Exception as e:
            print(f"Failed to initialize ZED: {str(e)}")
            self.zed_ready.set()
            return None

    def setup_video_writers(self, output_dir, frame_width, frame_height, is_zed=False):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 1
        writers = {}
        prefix = "ZED_" if is_zed else "RS_"
        writers["color"] = cv2.VideoWriter(f"{output_dir}/{prefix}ColorVideo.mp4", fourcc, fps,
                                           (frame_width, frame_height))
        writers["depth"] = cv2.VideoWriter(f"{output_dir}/{prefix}DepthVideo.mp4", fourcc, fps,
                                           (frame_width, frame_height))
        if is_zed:
            writers["left"] = cv2.VideoWriter(f"{output_dir}/{prefix}LeftVideo.mp4", fourcc, fps,
                                              (frame_width, frame_height))
            writers["right"] = cv2.VideoWriter(f"{output_dir}/{prefix}RightVideo.mp4", fourcc, fps,
                                               (frame_width, frame_height))
        else:
            writers["ir_left"] = cv2.VideoWriter(f"{output_dir}/{prefix}LeftIRVideo.mp4", fourcc, fps,
                                                 (frame_width, frame_height), isColor=False)
            writers["ir_right"] = cv2.VideoWriter(f"{output_dir}/{prefix}RightIRVideo.mp4", fourcc, fps,
                                                  (frame_width, frame_height), isColor=False)
        return writers

    def save_imu_data(self, imu_queue, output_file):
        imu_list = []
        while True:
            data = imu_queue.get()
            if data is None:
                break
            imu_list.append(data)
        if imu_list:
            pd.DataFrame(imu_list).to_csv(output_file, index=False)

    def record_realsense(self):
        global start_time
        pipeline, align = self.initialize_realsense()
        if pipeline is None or align is None:
            return

        self.zed_ready.wait()

        writers = self.setup_video_writers(self.realsense_data_dir, 1280, 720)
        imu_queue = queue.Queue()
        imu_thread = threading.Thread(target=self.save_imu_data,
                                      args=(imu_queue, f"{self.realsense_data_dir}/imu_data.csv"), daemon=True)
        imu_thread.start()

        frame_count = 0
        position = [0.0, 0.0, 0.0]
        velocity = [0.0, 0.0, 0.0]
        last_timestamp = None

        while self.running:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                frames = align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                gyro_frame = frames.first_or_default(rs.stream.gyro)
                accel_frame = frames.first_or_default(rs.stream.accel)
                ir_left_frame = frames.get_infrared_frame(1)
                ir_right_frame = frames.get_infrared_frame(2)

                if not color_frame or not depth_frame:
                    continue

                frame_count += 1
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255 / 5), cv2.COLORMAP_JET)

                writers["color"].write(color_image)
                writers["depth"].write(depth_colormap)

                if ir_left_frame and ir_right_frame:
                    ir_left = np.asanyarray(ir_left_frame.get_data())
                    ir_right = np.asanyarray(ir_right_frame.get_data())
                    writers["ir_left"].write(ir_left)
                    writers["ir_right"].write(ir_right)

                try:
                    self.realsense_frame_queue.put_nowait((color_image, depth_colormap))
                except queue.Full:
                    pass

                if gyro_frame and accel_frame:
                    gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                    accel_data = accel_frame.as_motion_frame().get_motion_data()
                    current_time = time.time_ns() - start_time  # Relative timestamp in ns
                    if last_timestamp:
                        dt = (current_time - last_timestamp) / 1e9  # Convert to seconds for physics
                        for i in range(3):
                            velocity[i] += getattr(accel_data, ['x', 'y', 'z'][i]) * dt
                            position[i] += velocity[i] * dt
                    last_timestamp = current_time

                    imu_queue.put({
                        "timestamp": current_time,
                        "position_x": position[0], "position_y": position[1], "position_z": position[2],
                        "acceleration_x": accel_data.x, "acceleration_y": accel_data.y, "acceleration_z": accel_data.z,
                        "angular_velocity_x": gyro_data.x, "angular_velocity_y": gyro_data.y,
                        "angular_velocity_z": gyro_data.z
                    })
            except RuntimeError as e:
                print(f"RealSense frame capture failed: {str(e)}")
                continue

        imu_queue.put(None)
        for writer in writers.values():
            writer.release()
        pipeline.stop()
        imu_thread.join()
        if frame_count > 0:
            elapsed_time = (time.time_ns() - start_time) / 1e9  # Convert ns to seconds
            fps = frame_count / elapsed_time
            print(f"RealSense: Completed in {elapsed_time:.2f} sec, FPS: {fps:.2f}")

    def record_zed(self):
        global start_time
        zed = self.initialize_zed()
        if zed is None:
            return

        self.realsense_ready.wait()

        writers = self.setup_video_writers(self.zed_data_dir,
                                           zed.get_camera_information().camera_configuration.resolution.width,
                                           zed.get_camera_information().camera_configuration.resolution.height, True)
        imu_queue = queue.Queue()
        imu_thread = threading.Thread(target=self.save_imu_data, args=(imu_queue, f"{self.zed_data_dir}/imu_data.csv"),
                                      daemon=True)
        imu_thread.start()

        frame_count = 0
        runtime_params = sl.RuntimeParameters()
        image_left, image_right, depth = sl.Mat(), sl.Mat(), sl.Mat()
        imu_data, pose = sl.SensorsData(), sl.Pose()

        while self.running:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                frame_count += 1
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

                # left_frame = cv2.cvtColor(image_left.get_data()[:, :, :3], cv2.COLOR_BGR2RGB)
                # right_frame = cv2.cvtColor(image_right.get_data()[:, :, :3], cv2.COLOR_BGR2RGB)
                left_frame = image_left.get_data()[:, :, :3]
                right_frame = image_right.get_data()[:, :, :3]
                depth_map = np.nan_to_num(depth.get_data(), nan=0)
                depth_display = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=255 / 5), cv2.COLORMAP_JET)

                writers["color"].write(left_frame)
                writers["left"].write(left_frame)
                writers["right"].write(right_frame)
                writers["depth"].write(depth_display)

                try:
                    self.zed_frame_queue.put_nowait((left_frame, depth_display))
                except queue.Full:
                    pass

                zed.get_sensors_data(imu_data, sl.TIME_REFERENCE.CURRENT)
                imu_values = imu_data.get_imu_data()
                accel = imu_values.get_linear_acceleration()
                gyro = imu_values.get_angular_velocity()
                zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                pos = pose.get_translation().get()

                imu_queue.put({
                    "timestamp": time.time_ns() - start_time,  # Relative timestamp in ns
                    "position_x": pos[0], "position_y": pos[1], "position_z": pos[2],
                    "acceleration_x": accel[0], "acceleration_y": accel[1], "acceleration_z": accel[2],
                    "angular_velocity_x": gyro[0], "angular_velocity_y": gyro[1], "angular_velocity_z": gyro[2]
                })

        imu_queue.put(None)
        for writer in writers.values():
            writer.release()
        zed.close()
        imu_thread.join()
        if frame_count > 0:
            elapsed_time = (time.time_ns() - start_time) / 1e9  # Convert ns to seconds
            fps = frame_count / elapsed_time
            print(f"ZED: Completed in {elapsed_time:.2f} sec, FPS: {fps:.2f}")

    def display_frames(self):
        while self.running:
            try:
                rs_color, rs_depth = self.realsense_frame_queue.get(timeout=1)
                cv2.imshow("RealSense Color", rs_color)
                # cv2.imshow("RealSense Depth", rs_depth)
            except queue.Empty:
                pass

            try:
                zed_color, zed_depth = self.zed_frame_queue.get(timeout=1)
                cv2.imshow("ZED Color", zed_color)
                # cv2.imshow("ZED Depth", zed_depth)
            except queue.Empty:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break

    def run(self):
        self.running = True
        rs_thread = threading.Thread(target=self.record_realsense)
        zed_thread = threading.Thread(target=self.record_zed)
        display_thread = threading.Thread(target=self.display_frames)

        print("Initializing cameras... Please wait")
        rs_thread.start()
        zed_thread.start()

        self.realsense_ready.wait()
        self.zed_ready.wait()

        if not (self.realsense_ready.is_set() and self.zed_ready.is_set()):
            print("One or both cameras failed to initialize. Shutting down...")
            self.running = False
        else:
            print("Both cameras ready! Recording started. Press 'q' in any window to stop")
            display_thread.start()

        rs_thread.join()
        zed_thread.join()
        display_thread.join()
        cv2.destroyAllWindows()
        print("✅ Data collection completed!")


if __name__ == "__main__":
    recorder = CameraRecorder()
    recorder.run()
