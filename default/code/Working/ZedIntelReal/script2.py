import argparse
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
import pyzed.sl as sl


class CameraRecorder:
    def __init__(self):
        self.running = False
        self.args = self.parse_args()
        self.zed_ready = threading.Event()
        self.realsense_ready = threading.Event()
        self.realsense_data_dir = os.path.join(self.args.output_dir, "RealSense_D435i")
        self.zed_data_dir = os.path.join(self.args.output_dir, "ZED2i_Data")
        os.makedirs(self.zed_data_dir, exist_ok=True)
        self.zed_frame_queue = queue.Queue(maxsize=10)
        os.makedirs(self.realsense_data_dir, exist_ok=True)
        self.realsense_frame_queue = queue.Queue(maxsize=10)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Dual Camera Data Collection with CSV logging")
        parser.add_argument("--output_dir", type=str, default="./Data2", help="Base output directory")
        parser.add_argument("--resolution", type=str, default="HD1080", choices=["HD720", "HD1080", "HD2K"],
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
            config.enable_stream(rs.stream.infrared, 0, 640, 480, rs.format.y8, 30)
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            config.enable_stream(rs.stream.gyro)
            config.enable_stream(rs.stream.accel)
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

    def save_csv(self, csv_path, data_list):
        if data_list:
            pd.DataFrame(data_list).to_csv(csv_path, index=False)

    def record_realsense(self):
        pipeline, align = self.initialize_realsense()
        if pipeline is None or align is None:
            return

        self.zed_ready.wait()
        log_data = []

        frame_count = 0
        start_time = time.time()

        while self.running:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                frames = align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                gyro_frame = frames.first_or_default(rs.stream.gyro)
                accel_frame = frames.first_or_default(rs.stream.accel)
                ir_left_frame = frames.get_infrared_frame(0)
                ir_right_frame = frames.get_infrared_frame(1)
                ir_left_file = f"intel_ir_left_{ts_str}.png"
                ir_right_file = f"intel_ir_right_{ts_str}.png"
                if ir_left_frame:
                    ir_left = np.asanyarray(ir_left_frame.get_data())
                    cv2.imwrite(os.path.join(self.realsense_data_dir, ir_left_file), ir_left)

                if ir_right_frame:
                    ir_right = np.asanyarray(ir_right_frame.get_data())
                    cv2.imwrite(os.path.join(self.realsense_data_dir, ir_right_file), ir_right)

                if not color_frame or not depth_frame:
                    continue

                frame_count += 1
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                # Save color image
                color_image = np.asanyarray(color_frame.get_data())
                color_filename = f"intel_color_{ts_str}.png"
                cv2.imwrite(os.path.join(self.realsense_data_dir, color_filename), color_image)

                # Save depth colormap
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255 / 5), cv2.COLORMAP_JET)
                depth_filename = f"intel_depth_{ts_str}.png"
                cv2.imwrite(os.path.join(self.realsense_data_dir, depth_filename), depth_colormap)

                # Display queue
                try:
                    self.realsense_frame_queue.put_nowait((color_image, depth_colormap))
                except queue.Full:
                    pass

                if gyro_frame and accel_frame:
                    gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                    accel_data = accel_frame.as_motion_frame().get_motion_data()

                    log_data.append({
                        "timestamp": ts_str,
                        "color_file": color_filename,
                        "depth_file": depth_filename,
                        "ir_left_file": ir_left_file,
                        "ir_right_file": ir_right_file,
                        "accel_x": accel_data.x, "accel_y": accel_data.y, "accel_z": accel_data.z,
                        "gyro_x": gyro_data.x, "gyro_y": gyro_data.y, "gyro_z": gyro_data.z
                    })

            except RuntimeError as e:
                print(f"RealSense frame capture failed: {str(e)}")
                continue

        pipeline.stop()
        self.save_csv(os.path.join(self.realsense_data_dir, "intel_log.csv"), log_data)

        if frame_count > 0:
            fps = frame_count / (time.time() - start_time)
            print(f"RealSense: Completed {frame_count} frames in {time.time() - start_time:.2f} sec, FPS: {fps:.2f}")

    def record_zed(self):
        zed = self.initialize_zed()
        if zed is None:
            return

        self.realsense_ready.wait()
        log_data = []

        frame_count = 0
        start_time = time.time()
        runtime_params = sl.RuntimeParameters()
        image_left, image_right, depth = sl.Mat(), sl.Mat(), sl.Mat()
        imu_data, pose = sl.SensorsData(), sl.Pose()

        while self.running:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                frame_count += 1
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

                left_frame = cv2.cvtColor(image_left.get_data()[:, :, :3], cv2.COLOR_RGBA2BGR)
                right_frame = cv2.cvtColor(image_right.get_data()[:, :, :3], cv2.COLOR_RGB2BGR)
                depth_map = np.nan_to_num(depth.get_data(), nan=0)
                depth_display = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=255 / 5), cv2.COLORMAP_JET)

                # Save images
                left_filename = f"zed_left_{ts_str}.png"
                right_filename = f"zed_right_{ts_str}.png"
                depth_filename = f"zed_depth_{ts_str}.png"
                cv2.imwrite(os.path.join(self.zed_data_dir, left_filename), left_frame)
                cv2.imwrite(os.path.join(self.zed_data_dir, right_filename), right_frame)
                cv2.imwrite(os.path.join(self.zed_data_dir, depth_filename), depth_display)

                # Display queue
                try:
                    self.zed_frame_queue.put_nowait((left_frame, depth_display))
                except queue.Full:
                    pass

                zed.get_sensors_data(imu_data, sl.TIME_REFERENCE.CURRENT)
                imu_values = imu_data.get_imu_data()
                accel = imu_values.get_linear_acceleration()
                gyro = imu_values.get_angular_velocity()

                log_data.append({
                    "timestamp": ts_str,
                    "left_file": left_filename,
                    "right_file": right_filename,
                    "depth_file": depth_filename,
                    "accel_x": accel[0], "accel_y": accel[1], "accel_z": accel[2],
                    "gyro_x": gyro[0], "gyro_y": gyro[1], "gyro_z": gyro[2]
                })

        zed.close()
        self.save_csv(os.path.join(self.zed_data_dir, "zed_log.csv"), log_data)

        if frame_count > 0:
            fps = frame_count / (time.time() - start_time)
            print(f"ZED: Completed {frame_count} frames in {time.time() - start_time:.2f} sec, FPS: {fps:.2f}")

    def display_frames(self):
        while self.running:
            try:
                rs_color, rs_depth = self.realsense_frame_queue.get(timeout=1)
                cv2.imshow("RealSense Color", rs_color)
                cv2.imshow("RealSense Depth", rs_depth)
            except queue.Empty:
                pass

            try:
                zed_color, zed_depth = self.zed_frame_queue.get(timeout=1)
                cv2.imshow("ZED Left", zed_color)
                cv2.imshow("ZED Depth", zed_depth)
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
