import cv2
import numpy as np
import glob
import os

# Checkerboard settings
CHECKERBOARD_SIZE = (10, 7)  # (Columns, Rows) of inner corners
SQUARE_SIZE = 0.025  # Set this to the actual square size in meters (e.g., 25mm = 0.025m)

# Path to calibration images
IMAGE_FOLDER = "calibration_images"

# Get sorted image file names
left_images = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "left_*.png")))
right_images = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "right_*.png")))

# Ensure both cameras have the same number of images
assert len(left_images) == len(right_images), "Number of left and right images must match!"

# Prepare object points (3D points)
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # Scale by square size

# Lists to store detected points
objpoints = []  # 3D world points
imgpoints_left = []  # 2D points from left camera
imgpoints_right = []  # 2D points from right camera

# Load images and detect corners
for left_img, right_img in zip(left_images, right_images):
    imgL = cv2.imread(left_img)
    imgR = cv2.imread(right_img)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find checkerboard corners
    foundL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD_SIZE, None)
    foundR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD_SIZE, None)

    if foundL and foundR:
        # Refine corner detection
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        # Store points
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
    else:
        print(f"Skipping pair {left_img} & {right_img}: Checkerboard not detected.")

# Get image size from first image
image_shape = grayL.shape[::-1]

# Calibrate left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, image_shape, None, None)
# Calibrate right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, image_shape, None, None)

# Stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC  # Keep individual camera calibrations fixed
retval, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, mtxL, distL, mtxR, distR, 
    image_shape, criteria=criteria, flags=flags
)

# Stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtxL, distL, mtxR, distR, image_shape, R, T)

# Compute undistortion maps for rectification
mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, image_shape, cv2.CV_16SC2)
mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, image_shape, cv2.CV_16SC2)

# Save calibration results
np.savez("stereo_calibration_data.npz", mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T, E=E, F=F, Q=Q, mapL1=mapL1, mapL2=mapL2, mapR1=mapR1, mapR2=mapR2)

print("✅ Stereo calibration complete!")
print(f"Reprojection Error: {retval:.4f}")
