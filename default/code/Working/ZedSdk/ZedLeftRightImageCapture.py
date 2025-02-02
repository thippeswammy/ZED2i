import time

import cv2
import pyzed.sl as sl

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
i = 0
print("press 's' for saving images")

cv2.namedWindow("Right Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Left Image", cv2.WINDOW_NORMAL)
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
        # ... (your processing code here)
        cv2.imshow("Left Image", image_left_cv)
        cv2.imshow("Right Image", image_right_cv)

        key = cv2.waitKey(1)
        if key == ord('s'):
            filename_left = f"left_image_{i}.jpg"
            filename_right = f"right_image_{i}.jpg"

            # Save right image
            cv2.imwrite(filename_left, image_left_cv)
            cv2.imwrite(filename_right, image_right_cv)
            print("=" * 100)
            i = i + 1
        elif key == ord('q'):
            break

# Close the camera
cam.close()
cv2.destroyAllWindows()

# In two frame it is showing same images
