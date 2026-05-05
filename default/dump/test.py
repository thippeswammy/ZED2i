import cv2
import numpy as np
import pyzed.sl as sl
import time


# Initialize Camera
def initialize_camera():
    print("\n🔹 [TEST 1] Initializing Camera...")
    camera = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER

    err = camera.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"❌ Camera failed to open: {err}")
        exit()
    print("✅ Camera successfully opened!\n")
    return camera


# Get Camera Information
def get_camera_info(camera):
    print("\n🔹 [TEST 2] Fetching Camera Info...")
    info = camera.get_camera_information()

    print(f"✅ ZED Model         : {info.camera_model}")
    print(f"✅ Serial Number     : {info.serial_number}")

    # ✅ FIX: Get firmware version correctly
    firmware_version = camera.get_camera_firmware_version()
    print(f"✅ Firmware Version  : {firmware_version}")

    resolution = info.camera_configuration.resolution
    print(f"✅ Camera Resolution : {resolution.width}x{resolution.height}\n")





# Capture and Display Image
def capture_image(camera):
    print("\n🔹 [TEST 3] Capturing Image...")
    image = sl.Mat()

    if camera.grab() == sl.ERROR_CODE.SUCCESS:
        camera.retrieve_image(image, sl.VIEW.LEFT)
        img_np = image.get_data()
        cv2.imshow("ZED 2i Camera Frame", img_np)
        cv2.waitKey(3000)  # Show image for 3 seconds
        cv2.destroyAllWindows()
        print("✅ Image Captured Successfully!\n")
    else:
        print("❌ Image Capture Failed!\n")


# Depth Sensing Test
def test_depth_sensing(camera):
    print("\n🔹 [TEST 4] Testing Depth Sensing...")
    depth = sl.Mat()

    if camera.grab() == sl.ERROR_CODE.SUCCESS:
        camera.retrieve_measure(depth, sl.MEASURE.DEPTH)
        print("✅ Depth data retrieved successfully!\n")
    else:
        print("❌ Failed to retrieve depth data!\n")


# Body Tracking Test
def test_body_tracking(camera):
    print("\n🔹 [TEST 5] Testing Body Tracking...")
    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking = True
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    camera.enable_body_tracking(body_params)

    bodies = sl.Bodies()
    if camera.grab() == sl.ERROR_CODE.SUCCESS:
        camera.retrieve_bodies(bodies)
        print(f"✅ Detected {len(bodies.body_list)} people.\n")
    else:
        print("❌ Body Tracking Failed!\n")


# Self-Calibration Check
def check_self_calibration(camera):
    print("\n🔹 [TEST 6] Checking Self-Calibration...")
    if camera.is_self_calibration_enabled():
        print("✅ Self-Calibration is Enabled.")
    else:
        print("❌ Self-Calibration is Disabled. Try adjusting lighting and surroundings.\n")


# FPS & Performance Test
def test_fps_performance(camera):
    print("\n🔹 [TEST 7] Measuring Camera FPS...")
    start_time = time.time()
    frames = 0

    while time.time() - start_time < 5:  # Test for 5 seconds
        if camera.grab() == sl.ERROR_CODE.SUCCESS:
            frames += 1

    fps = frames / 5.0
    print(f"✅ Measured FPS: {fps:.2f}\n")


# Run All Tests
def run_all_tests():
    camera = initialize_camera()
    get_camera_info(camera)
    capture_image(camera)
    test_depth_sensing(camera)
    test_body_tracking(camera)
    check_self_calibration(camera)
    test_fps_performance(camera)

    camera.close()
    print("🎉 All Tests Completed!")


# Execute tests
if __name__ == "__main__":
    run_all_tests()
