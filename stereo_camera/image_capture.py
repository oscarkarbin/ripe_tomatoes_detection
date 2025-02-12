import cv2
import numpy as np
import os
import time

# Camera IDs (Change if needed)
CAMERA_LEFT = 1  # Adjust based on your setup
CAMERA_RIGHT = 2  # Adjust based on your setup

# Checkerboard settings
CHECKERBOARD_SIZE = (10, 7)  # Number of inner corners (vertices)
SAVE_FOLDER = "calibration_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Open cameras
cap_left = cv2.VideoCapture(CAMERA_LEFT)
cap_right = cv2.VideoCapture(CAMERA_RIGHT)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error: One or both cameras could not be opened.")
    exit()

image_count = 0
last_corners_left = None
last_corners_right = None
last_capture_time = time.time()

while True:
    # Capture frames
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Error: Unable to read frames from cameras.")
        break

    # Convert to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Find checkerboard corners
    found_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD_SIZE, None)
    found_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD_SIZE, None)



    # Display camera feeds
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)

    # Auto-capture logic
    if found_left and found_right:
        # Compute movement by comparing new and old corner positions
        if last_corners_left is not None and last_corners_right is not None:
            movement_left = np.linalg.norm(corners_left - last_corners_left)
            movement_right = np.linalg.norm(corners_right - last_corners_right)
        else:
            movement_left, movement_right = float('inf'), float('inf')  # First capture is always allowed

        # If the board is steady and enough time has passed, capture the images
        if movement_left < 3.0 and movement_right < 3.0 and (time.time() - last_capture_time) > 2:
            img_name_left = f"{SAVE_FOLDER}/left_{image_count:02d}.png"
            img_name_right = f"{SAVE_FOLDER}/right_{image_count:02d}.png"

            cv2.imwrite(img_name_left, frame_left)
            cv2.imwrite(img_name_right, frame_right)

            print(f"Captured: {img_name_left} & {img_name_right}")

            image_count += 1
            last_capture_time = time.time()  # Update last capture time

        # Store previous corners for movement detection
        last_corners_left = corners_left.copy()
        last_corners_right = corners_right.copy()

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
