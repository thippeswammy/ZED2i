import cv2
import pyzed.sl as sl

i = 0
init = sl.InitParameters()
cam = sl.Camera()

# Open the camera
status = cam.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print("Camera Open : " + repr(status) + ". Exit program.")
    exit()

# Create a sl.Mat object to store retrieved images
image_zedL = sl.Mat()
image_zedR = sl.Mat()

cam_info = cam.get_camera_information()
print("=" * 60)
print("ZED Camera Resolution     : {0}x{1}".format(round(cam_info.camera_configuration.resolution.width, 2),
                                                   cam.get_camera_information().camera_configuration.resolution.height))
print("=" * 60)
while True:
    # Grab a new image frame
    runtime_parameters = sl.RuntimeParameters()
    if cam.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

        # Retrieve the left image ,Retrieve the right image
        cam.retrieve_image(image_zedL, sl.VIEW.LEFT)
        cam.retrieve_image(image_zedR, sl.VIEW.RIGHT)

        image_left_cv = image_zedL.get_data()  # Convert to OpenCV Mat
        image_right_cv = image_zedR.get_data()  # Convert to OpenCV Mat

        # Process or display the left and right images using OpenCV
        cv2.imshow("Left Image", image_left_cv)
        cv2.imshow("Right Image", image_right_cv)

        # 's' for Save Images
        key = cv2.waitKey(1)
        if key == ord('s'):
            filename_left = f"left_image_{i}.png"
            filename_right = f"right_image_{i}.png"

            cv2.imwrite(filename_left, image_left_cv)
            cv2.imwrite(filename_right, image_right_cv)
            print("=" * 50)
            i = i + 1

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

# Close the camera
cam.close()
cv2.destroyAllWindows()
