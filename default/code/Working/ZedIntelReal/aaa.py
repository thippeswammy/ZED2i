import pyrealsense2 as rs
import numpy as np
import cv2

def record_avi(filename="output.avi"):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream at 1920x1080, 30 fps
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (1920, 1080))

    print(f"Recording to {filename}. Press 'q' to quit.")

    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Write the frame
            out.write(color_image)

            # Show the frame
            cv2.namedWindow('RealSense 1080p', cv2.WINDOW_NORMAL)
            cv2.imshow('RealSense 1080p', color_image)

            # Press 'q' to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Stop streaming and release resources
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    record_avi()