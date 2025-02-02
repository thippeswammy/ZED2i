import time
import cv2
import pyzed.sl as sl

# Initialize camera
init = sl.InitParameters()
cam = sl.Camera()

# Open the camera
status = cam.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print("Camera Open : " + repr(status) + ". Exit program.")
    exit()

# Create sl.Mat objects to store retrieved images
image_zedL = sl.Mat()
image_zedR = sl.Mat()
i = 0
print("Press 's' to save images and 'q' to quit")

while True:
    # Grab a new frame
    runtime_parameters = sl.RuntimeParameters()
    if cam.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left and right images from the ZED camera
        cam.retrieve_image(image_zedL, sl.VIEW.LEFT)
        cam.retrieve_image(image_zedR, sl.VIEW.RIGHT)

        # Convert ZED images to numpy arrays for OpenCV
        image_left_cv = image_zedL.get_data()
        image_right_cv = image_zedR.get_data()

        # Display the images using OpenCV
        cv2.imshow("Left Image", image_left_cv)
        cv2.imshow("Right Image", image_right_cv)

        # Key press events
        key = cv2.waitKey(1)  # Wait for 1 ms and capture key press
        if key == ord('s'):  # Save images if 's' is pressed
            filename_left = f"left_image_{i}.png"
            filename_right = f"right_image_{i}.png"
            image_zedL.write(filename_left)
            image_zedR.write(filename_right)
            print(f"Saved: {filename_left} and {filename_right}")
            i += 1
        elif key == ord('q'):  # Quit if 'q' is pressed
            break

# Release resources
cam.close()
cv2.destroyAllWindows()  # Close all OpenCV windows
