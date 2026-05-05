# ########################################################################
# #
# # Copyright (c) 2022, STEREOLABS.
# #
# # All rights reserved.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# # "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# # LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# # A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# # OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# # SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# # LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# # DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# # THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# #
# ########################################################################
#
# """
#     Live camera sample showing the camera information and video in real time and allows to control the different
#     settings.
# """
#
# import cv2
# import pyzed.sl as sl
#
# # Global variable
# camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
# str_camera_settings = "BRIGHTNESS"
# step_camera_settings = 1
# led_on = True
# selection_rect = sl.Rect()
# select_in_progress = False
# origin_rect = (-1, -1)
#
#
# # Function that handles mouse events when interacting with the OpenCV window.
# def on_mouse(event, x, y, flags, param):
#     global select_in_progress, selection_rect, origin_rect
#     if event == cv2.EVENT_LBUTTONDOWN:
#         origin_rect = (x, y)
#         selection_rect.x = x
#         selection_rect.y = y
#         select_in_progress = True
#     elif event == cv2.EVENT_LBUTTONUP:
#         select_in_progress = False
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         select_in_progress = False
#         selection_rect = sl.Rect(0, 0, 0, 0)
#
#     if select_in_progress:
#         # selection_rect.x = max(min(x, origin_rect[0]),0)
#         # selection_rect.y = max(min(y, origin_rect[1]),0)
#         selection_rect.width = max(x, 0)
#         selection_rect.height = max(y, 0)
#
#         # selection_rect.x = min(x, origin_rect[0])
#         # selection_rect.y = min(y, origin_rect[1])
#         # selection_rect.width = abs(x - origin_rect[0]) + 1
#         # selection_rect.height = abs(y - origin_rect[1]) + 1
#
#
# import time
#
#
# def main():
#     WIDTH = 1280
#     HIGHT = 720
#     init = sl.InitParameters()
#     init.camera_fps = 60
#     cam = sl.Camera()
#     status = cam.open(init)
#     if status != sl.ERROR_CODE.SUCCESS:
#         print("Camera Open : " + repr(status) + ". Exit program.")
#         exit()
#
#     runtime = sl.RuntimeParameters()
#     mat = sl.Mat()
#     win_name = "Camera Control"
#     cv2.namedWindow(win_name)
#     cv2.setMouseCallback(win_name, on_mouse)
#     print_camera_information(cam)
#     print_help()
#     switch_camera_settings()
#
#     # --- FPS variables ---
#     prev_time = time.time()
#     fps = 0
#     frame_count = 0
#
#     key = ''
#     while key != 113:  # for 'q' key
#         err = cam.grab(runtime)
#         if err == sl.ERROR_CODE.SUCCESS:
#             cam.retrieve_image(mat, sl.VIEW.LEFT)
#             cvImage = mat.get_data()
#
#             # --- FPS Calculation ---
#             frame_count += 1
#             if frame_count >= 10:  # update every 10 frames for stability
#                 now = time.time()
#                 fps = frame_count / (now - prev_time)
#                 prev_time = now
#                 frame_count = 0
#
#             # --- Draw FPS on frame ---
#             cv2.putText(cvImage, f"FPS: {fps:.2f}", (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
#
#             # Draw selection if active
#             selection_rect.width = max(min(selection_rect.width, WIDTH), 1)
#             selection_rect.height = max(min(selection_rect.height, HIGHT), 1)
#
#             if not selection_rect.is_empty():
#                 cv2.rectangle(cvImage,
#                               (min(selection_rect.x, selection_rect.width),
#                                min(selection_rect.y, selection_rect.height)),
#                               (max(selection_rect.x, selection_rect.width),
#                                max(selection_rect.y, selection_rect.height)),
#                               (220, 180, 20), 2)
#
#             cv2.imshow(win_name, cvImage)
#         else:
#             print("Error during capture : ", err)
#             break
#
#         key = cv2.waitKey(5)
#         update_camera_settings(key, cam, runtime, mat)
#
#     cv2.destroyAllWindows()
#     cam.close()
#
#
# # Display camera information
# def print_camera_information(cam):
#     cam_info = cam.get_camera_information()
#     print("ZED Model                 : {0}".format(cam_info.camera_model))
#     print("ZED Serial Number         : {0}".format(cam_info.serial_number))
#     print("ZED Camera Firmware       : {0}/{1}".format(cam_info.camera_configuration.firmware_version,
#                                                        cam_info.sensors_configuration.firmware_version))
#     print("ZED Camera Resolution     : {0}x{1}".format(round(cam_info.camera_configuration.resolution.width, 2),
#                                                        cam.get_camera_information().camera_configuration.resolution.height))
#     print("ZED Camera FPS            : {0}".format(int(cam_info.camera_configuration.fps)))
#
#
# # Print help
# def print_help():
#     print("\n\nCamera controls hotkeys:")
#     print("* Increase camera settings value:  '+'")
#     print("* Decrease camera settings value:  '-'")
#     print("* Toggle camera settings:          's'")
#     print("* Toggle camera LED:               'l' (lower L)")
#     print("* Reset all parameters:            'r'")
#     print("* Reset exposure ROI to full image 'f'")
#     print("* Use mouse to select an image area to apply exposure (press 'a')")
#     print("* Exit :                           'q'\n")
#
#
# # update camera setting on key press
# def update_camera_settings(key, cam, runtime, mat):
#     global led_on
#     if key == 115:  # for 's' key
#         # Switch camera settings
#         switch_camera_settings()
#     elif key == 43:  # for '+' key
#         # Increase camera settings value.
#         current_value = cam.get_camera_settings(camera_settings)[1]
#         cam.set_camera_settings(camera_settings, current_value + step_camera_settings)
#         print(str_camera_settings + ": " + str(current_value + step_camera_settings))
#     elif key == 45:  # for '-' key
#         # Decrease camera settings value.
#         current_value = cam.get_camera_settings(camera_settings)[1]
#         if current_value >= 1:
#             cam.set_camera_settings(camera_settings, current_value - step_camera_settings)
#             print(str_camera_settings + ": " + str(current_value - step_camera_settings))
#     elif key == 114:  # for 'r' key
#         # Reset all camera settings to default.
#         cam.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, -1)
#         cam.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, -1)
#         cam.set_camera_settings(sl.VIDEO_SETTINGS.HUE, -1)
#         cam.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, -1)
#         cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, -1)
#         cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)
#         cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
#         cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)
#         print("[Sample] Reset all settings to default")
#     elif key == 108:  # for 'l' key
#         # Turn on or off camera LED.
#         led_on = not led_on
#         cam.set_camera_settings(sl.VIDEO_SETTINGS.LED_STATUS, led_on)
#     elif key == 97:  # for 'a' key
#         # Set exposure region of interest (ROI) on a target area.
#         print("[Sample] set AEC_AGC_ROI on target [", selection_rect.x, ",", selection_rect.y, ",",
#               selection_rect.width, ",", selection_rect.height, "]")
#         cam.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, selection_rect, sl.SIDE.BOTH)
#     elif key == 102:  # for 'f' key
#         # Reset exposure ROI to full resolution.
#         print("[Sample] reset AEC_AGC_ROI to full res")
#         cam.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, selection_rect, sl.SIDE.BOTH, True)
#
#
# # Function to switch between different camera settings (brightness, contrast, etc.).
# def switch_camera_settings():
#     global camera_settings
#     global str_camera_settings
#     if camera_settings == sl.VIDEO_SETTINGS.BRIGHTNESS:
#         camera_settings = sl.VIDEO_SETTINGS.CONTRAST
#         str_camera_settings = "Contrast"
#         print("[Sample] Switch to camera settings: CONTRAST")
#     elif camera_settings == sl.VIDEO_SETTINGS.CONTRAST:
#         camera_settings = sl.VIDEO_SETTINGS.HUE
#         str_camera_settings = "Hue"
#         print("[Sample] Switch to camera settings: HUE")
#     elif camera_settings == sl.VIDEO_SETTINGS.HUE:
#         camera_settings = sl.VIDEO_SETTINGS.SATURATION
#         str_camera_settings = "Saturation"
#         print("[Sample] Switch to camera settings: SATURATION")
#     elif camera_settings == sl.VIDEO_SETTINGS.SATURATION:
#         camera_settings = sl.VIDEO_SETTINGS.SHARPNESS
#         str_camera_settings = "Sharpness"
#         print("[Sample] Switch to camera settings: Sharpness")
#     elif camera_settings == sl.VIDEO_SETTINGS.SHARPNESS:
#         camera_settings = sl.VIDEO_SETTINGS.GAIN
#         str_camera_settings = "Gain"
#         print("[Sample] Switch to camera settings: GAIN")
#     elif camera_settings == sl.VIDEO_SETTINGS.GAIN:
#         camera_settings = sl.VIDEO_SETTINGS.EXPOSURE
#         str_camera_settings = "Exposure"
#         print("[Sample] Switch to camera settings: EXPOSURE")
#     elif camera_settings == sl.VIDEO_SETTINGS.EXPOSURE:
#         camera_settings = sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE
#         str_camera_settings = "White Balance"
#         print("[Sample] Switch to camera settings: WHITEBALANCE")
#     elif camera_settings == sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE:
#         camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
#         str_camera_settings = "Brightness"
#         print("[Sample] Switch to camera settings: BRIGHTNESS")
#
#
# if __name__ == "__main__":
#     main()

import csv
import time
import cv2
import pyzed.sl as sl


def main():
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.METER

    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Camera not opened")
        exit()

    runtime = sl.RuntimeParameters()
    left = sl.Mat()
    right = sl.Mat()
    depth = sl.Mat()
    imu_data = sl.SensorsData()

    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = cam.get_camera_information().camera_configuration.resolution.width
    height = cam.get_camera_information().camera_configuration.resolution.height
    fps = cam.get_camera_information().camera_configuration.fps

    left_writer = cv2.VideoWriter("left.avi", fourcc, fps, (width, height))
    right_writer = cv2.VideoWriter("right.avi", fourcc, fps, (width, height))

    # CSV for IMU
    imu_file = open("imu.csv", "w", newline="")
    imu_writer = csv.writer(imu_file)
    imu_writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz"])

    print("Recording streams... Press 'q' to quit.")

    # FPS calculation variables
    prev_time = time.time()
    frame_count = 0
    fps_display = 0

    key = ''
    while key != 113:  # 'q'
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # Retrieve images
            cam.retrieve_image(left, sl.VIEW.LEFT)
            cam.retrieve_image(right, sl.VIEW.RIGHT)
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH)

            # Convert to OpenCV format (BGR for saving)
            left_bgr = cv2.cvtColor(left.get_data(), cv2.COLOR_BGRA2BGR)
            right_bgr = cv2.cvtColor(right.get_data(), cv2.COLOR_BGRA2BGR)

            # Save video
            left_writer.write(left_bgr)
            right_writer.write(right_bgr)

            # --- FPS Calculation ---
            frame_count += 1
            if frame_count >= 10:  # update every 10 frames
                now = time.time()
                fps_display = frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0

            # Overlay FPS on Left image
            cv2.putText(left_bgr, f"FPS: {fps_display:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Show all windows
            cv2.imshow("Left (with FPS)", left_bgr)
            cv2.imshow("Right", right_bgr)

            # Depth is grayscale 32F -> convert for display
            depth_display = depth.get_data()
            depth_display = cv2.convertScaleAbs(depth_display, alpha=0.05)  # normalize for visibility
            cv2.imshow("Depth", depth_display)

            # IMU logging
            if cam.get_sensors_data(imu_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                imu = imu_data.get_imu_data()
                a = imu.get_linear_acceleration()
                g = imu.get_angular_velocity()
                ts = imu.timestamp.get_milliseconds()
                imu_writer.writerow([ts, a[0], a[1], a[2], g[0], g[1], g[2]])

            key = cv2.waitKey(1)

    # Cleanup
    left_writer.release()
    right_writer.release()
    imu_file.close()
    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    main()
