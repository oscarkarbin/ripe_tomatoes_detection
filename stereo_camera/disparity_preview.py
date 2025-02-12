import cv2
import numpy as np

# Load calibration data
calib_data = np.load("stereo_calibration_data.npz")
mtxL, distL = calib_data["mtxL"], calib_data["distL"]
mtxR, distR = calib_data["mtxR"], calib_data["distR"]
mapL1, mapL2, mapR1, mapR2 = calib_data["mapL1"], calib_data["mapL2"], calib_data["mapR1"], calib_data["mapR2"]

# Camera IDs (adjust if needed)
CAMERA_LEFT = 1
CAMERA_RIGHT = 2

# Set resolution
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Open cameras
capL = cv2.VideoCapture(CAMERA_LEFT)
capR = cv2.VideoCapture(CAMERA_RIGHT)

# Set resolution for both cameras
capL.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not capL.isOpened() or not capR.isOpened():
    print("Error: Could not open cameras.")
    exit()

# Stereo SGBM settings
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # Must be multiple of 16
    blockSize=9,
    P1=8 * 3 * 9**2,
    P2=32 * 3 * 9**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

while True:
    # Capture frames
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Error: Could not read frames.")
        break

    # Convert to grayscale
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Undistort and rectify
    rectifiedL = cv2.remap(grayL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(grayR, mapR1, mapR2, cv2.INTER_LINEAR)

    # Compute disparity
    disparity = stereo.compute(rectifiedL, rectifiedR).astype(np.float32) / 16.0

    # Normalize for display
    disparity_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_vis = np.uint8(disparity_vis)

    # Display windows
    cv2.imshow("Left Camera (Rectified)", rectifiedL)
    cv2.imshow("Right Camera (Rectified)", rectifiedR)
    cv2.imshow("Disparity Map", disparity_vis)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
capL.release()
capR.release()
cv2.destroyAllWindows()
