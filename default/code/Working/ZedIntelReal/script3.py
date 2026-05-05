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


class RealSenseRecorder:
    def __init__(self):
        self.running = False
        self.args = self.parse_args()
        self.realsense_ready = threading.Event()
        self.realsense_data_dir = os.path.join(self.args.output_dir, "RealSense_D435i")
        os.makedirs(self.realsense_data_dir, exist_ok=True)
        self.frame_queue = queue.Queue(maxsize=10)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Intel RealSense Data Collection with CSV logging")
        parser.add_argument("--output_dir", type=str, default="./Data_RealSense", help="Base output directory")
        parser.add_argument("--fps", type=int, default=30, help="Frames per second")
        return parser.parse_args()

    def initialize_realsense(self):
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            
            # Using 1920x1080 for RGB as requested, 1280x720 for Depth and IR
            color_w, color_h = 1080, 720
            depth_w, depth_h = 720, 480 
            fps = self.args.fps

            print(f"Configuring streams:")
            print(f"  - Color: {color_w}x{color_h} @ {fps} FPS")
            print(f"  - Depth/IR: {depth_w}x{depth_h} @ {fps} FPS")
            
            config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, fps)
            
            # Use infrared 1 and 2 (Left and Right) which is standard for D435i
            config.enable_stream(rs.stream.infrared, 1, depth_w, depth_h, rs.format.y8, fps)
            config.enable_stream(rs.stream.infrared, 2, depth_w, depth_h, rs.format.y8, fps)
            
            # IMU streams
            config.enable_stream(rs.stream.gyro)
            config.enable_stream(rs.stream.accel)
            
            pipeline.start(config)
            
            # Warmup
            print("Warming up sensors...")
            for _ in range(15):
                pipeline.wait_for_frames(timeout_ms=5000)
                
            self.realsense_ready.set()
            return pipeline, rs.align(rs.stream.color)
        except Exception as e:
            print(f"Failed to initialize RealSense: {str(e)}")
            print("Tip: 1920x1080 @ 30 FPS requires a high-quality USB 3.0 cable and port.")
            self.init_failed = True
            self.realsense_ready.set()
            return None, None

    def save_csv(self, csv_path, data_list):
        if data_list:
            pd.DataFrame(data_list).to_csv(csv_path, index=False)
            print(f"\n✅ Log saved to {csv_path}")

    def record_realsense(self):
        pipeline, align = self.initialize_realsense()
        if pipeline is None:
            self.running = False
            return

        log_data = []
        frame_count = 0
        start_time = time.time()

        while self.running:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=3000)
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
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                # Save images
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Colormap for depth visualization
                depth_display = cv2.convertScaleAbs(depth_image, alpha=0.03) 
                depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

                color_filename = f"intel_color_{ts_str}.png"
                depth_filename = f"intel_depth_{ts_str}.png"
                cv2.imwrite(os.path.join(self.realsense_data_dir, color_filename), color_image)
                cv2.imwrite(os.path.join(self.realsense_data_dir, depth_filename), depth_colormap)

                ir_left_file = "N/A"
                ir_right_file = "N/A"

                if ir_left_frame:
                    ir_left_file = f"intel_ir_left_{ts_str}.png"
                    cv2.imwrite(os.path.join(self.realsense_data_dir, ir_left_file), np.asanyarray(ir_left_frame.get_data()))

                if ir_right_frame:
                    ir_right_file = f"intel_ir_right_{ts_str}.png"
                    cv2.imwrite(os.path.join(self.realsense_data_dir, ir_right_file), np.asanyarray(ir_right_frame.get_data()))

                # Update preview queue
                try:
                    self.frame_queue.put_nowait((color_image, depth_colormap))
                except queue.Full:
                    pass

                # IMU logging
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

            except Exception as e:
                print(f"Capture error: {str(e)}")
                continue

        pipeline.stop()
        self.save_csv(os.path.join(self.realsense_data_dir, "intel_log.csv"), log_data)
        if frame_count > 0:
            print(f"Recorded {frame_count} frames. Avg FPS: {frame_count / (time.time() - start_time):.2f}")

    def display_frames(self):
        while self.running:
            try:
                color, depth = self.frame_queue.get(timeout=0.5)
                cv2.imshow("RealSense Color", color)
                cv2.imshow("RealSense Depth", depth)
            except queue.Empty:
                if not self.running: break
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break

    def run(self):
        self.running = True
        self.init_failed = False
        rs_thread = threading.Thread(target=self.record_realsense)
        
        print("Initializing RealSense... Please wait")
        rs_thread.start()

        # Wait for initialization
        self.realsense_ready.wait(timeout=15)

        if self.realsense_ready.is_set() and not self.init_failed:
            print("Recording started. Press 'q' to stop.")
            self.display_frames()
        else:
            print("Failed to initialize camera or timeout reached.")
            self.running = False

        rs_thread.join()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    recorder = RealSenseRecorder()
    recorder.run()
