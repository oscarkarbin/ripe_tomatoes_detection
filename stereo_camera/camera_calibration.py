import os
import glob
import cv2
import numpy as np

def find_chessboard_corners(image_paths, chessboard_size, square_size=1.0, display=False):
    """
    Finds chessboard corners for all images in 'image_paths'.

    :param image_paths: List of file paths to images.
    :param chessboard_size: (columns, rows) for the internal corners of the checkerboard.
    :param square_size: Physical size of each checkerboard square (in millimeters, if desired).
    :param display: If True, shows corner detection on screen for debugging.
    :return: (objpoints, imgpoints, image_size)
             objpoints  -> List of 3D points in the real world (for all images).
             imgpoints  -> List of 2D points in the image plane.
             image_size -> (width, height) of the images (assumed the same for all).
    """
    cols, rows = chessboard_size
    
    # Prepare a single array of the 3D points for the corners in the checkerboard
    # The Z coordinate is zero because it’s on a flat plane
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size  # Scale by the actual size of a square
    
    objpoints = []  # 3D points (same for each successful detection)
    imgpoints = []  # 2D corner points in each image
    
    image_size = None

    # Termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    for fpath in image_paths:
        img = cv2.imread(fpath)
        if img is None:
            print(f"Warning: could not read {fpath}. Skipping.")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Update image_size based on the first valid image
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        
        if ret:
            # Refine corners for better accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            
            # Optional: display corners for debugging
            if display:
                cv2.drawChessboardCorners(img, (cols, rows), corners_refined, ret)
                cv2.imshow('Chessboard', img)
                cv2.waitKey(500)
    
    if display:
        cv2.destroyAllWindows()
    
    return objpoints, imgpoints, image_size

def calibrate_single_camera(objpoints, imgpoints, image_size):
    """
    Calibrates a single camera given object points, image points, and image size.
    :return: cameraMatrix, distCoeffs, rvecs, tvecs
    """
    # Run single camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    if not ret:
        raise RuntimeError("Single camera calibration failed. Check your data.")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs

def stereo_calibrate(
    objpoints, 
    left_imgpoints,
    right_imgpoints,
    cameraMatrix1,
    distCoeffs1,
    cameraMatrix2,
    distCoeffs2,
    image_size
):
    """
    Performs stereo calibration given object points, corresponding left and right image points,
    and initial intrinsics/distortions for both cameras.
    
    :return: R, T, E, F  (rotation, translation, essential matrix, fundamental matrix)
    """
    # Stereo calibration flags
    flags = 0
    # Commonly, we fix the intrinsics from individual calibration. 
    # If you want to let stereoCalibrate optimize intrinsics too, comment this out.
    flags |= cv2.CALIB_FIX_INTRINSIC
    
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
    
    # Stereo calibrate
    retval, cm1, dc1, cm2, dc2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        left_imgpoints,
        right_imgpoints,
        cameraMatrix1,
        distCoeffs1,
        cameraMatrix2,
        distCoeffs2,
        image_size,  # (width, height)
        criteria=criteria,
        flags=flags
    )
    
    if retval < 0.1:  # Heuristic check; you can adjust or remove
        print("Warning: stereo calibration might not have converged well.")
    
    return R, T, E, F

def stereo_rectify(
    cameraMatrix1,
    distCoeffs1,
    cameraMatrix2,
    distCoeffs2,
    image_size,
    R,
    T
):
    """
    Computes the rectification transforms for both cameras to align stereo images.
    :return: (R1, R2, P1, P2, Q) for subsequent rectification.
    """
    # alpha=0 means we crop the image to avoid black areas, alpha=1 means no cropping
    alpha = 0
    
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1,
        distCoeffs1,
        cameraMatrix2,
        distCoeffs2,
        image_size,
        R,
        T,
        alpha=alpha
    )
    return R1, R2, P1, P2, Q

def compute_disparity_map(left_image, right_image, stereo_matcher):
    """
    Computes a disparity (depth) map from rectified stereo images.
    """
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    
    disparity = stereo_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    
    # Normalize for visualization
    disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return disparity, disparity_vis

def reproject_to_3D(disparity, Q):
    """
    Converts disparity map to 3D coordinates using the Q matrix.
    """
    return cv2.reprojectImageTo3D(disparity, Q)


def process_stereo_images(R1, R2, P1, P2, Q, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size):
    """
    Loads stereo images, rectifies them, computes disparity, and estimates depth.
    """
    capL = cv2.VideoCapture(1)  # Left camera
    capR = cv2.VideoCapture(2)  # Right camera

    capL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Generate rectification maps
    mapLx, mapLy = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, image_size, cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, image_size, cv2.CV_32FC1)

    # Stereo Block Matching
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,  # Must be multiple of 16
        blockSize=9,
        P1=8 * 3 * 9**2,
        P2=32 * 3 * 9**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()

        if not retL or not retR:
            print("Error: Could not capture frames.")
            break

        # Rectify frames
        rectifiedL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
        rectifiedR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

        for y in range(0, rectifiedL.shape[0], 50):  # Every 50 pixels
            cv2.line(rectifiedL, (0, y), (rectifiedL.shape[1], y), (0, 255, 0), 1)
            cv2.line(rectifiedR, (0, y), (rectifiedR.shape[1], y), (0, 255, 0), 1)


        # ✅ **Check rectification before computing disparity**
        cv2.imshow("Rectified Left", rectifiedL)
        cv2.imshow("Rectified Right", rectifiedR)
        cv2.waitKey(0)  # Press any key to continue
        cv2.destroyAllWindows()

        # Compute disparity map
        disparity, disparity_vis = compute_disparity_map(rectifiedL, rectifiedR, stereo)

        # Convert disparity to real-world depth
        depth_map = reproject_to_3D(disparity, Q)

        # Display
        cv2.imshow("Rectified Left", rectifiedL)
        cv2.imshow("Rectified Right", rectifiedR)
        cv2.imshow("Disparity Map", disparity_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()



def main(
    left_images_glob="left/*.jpg",
    right_images_glob="right/*.jpg",
    chessboard_size=(10, 7),
    square_size=25.0,  # Changed to 25mm per square
    display=False
):
    # 1) Collect all left and right image file paths
    left_image_paths = sorted(glob.glob(left_images_glob))
    right_image_paths = sorted(glob.glob(right_images_glob))
    
    if len(left_image_paths) != len(right_image_paths):
        print("Warning: Number of left and right images does not match.")
    
    if len(left_image_paths) == 0:
        raise ValueError("No left images found. Check path.")
    if len(right_image_paths) == 0:
        raise ValueError("No right images found. Check path.")

    # 2) Detect corners in both left and right images for the same poses
    all_objpoints = []
    left_imgpoints_final = []
    right_imgpoints_final = []
    
    image_size = None  # We'll get this after we read the first valid pair

    # Prepare base object points for the known chessboard pattern
    cols, rows = chessboard_size
    base_objp = np.zeros((rows * cols, 3), np.float32)
    base_objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    base_objp *= square_size

    # Corner refinement criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for left_f, right_f in zip(left_image_paths, right_image_paths):
        left_img = cv2.imread(left_f)
        right_img = cv2.imread(right_f)
        if left_img is None or right_img is None:
            continue
        
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (left_gray.shape[1], left_gray.shape[0])
        
        # Find corners
        ret_left, corners_left = cv2.findChessboardCorners(left_gray, (cols, rows), None)
        ret_right, corners_right = cv2.findChessboardCorners(right_gray, (cols, rows), None)
        
        if ret_left and ret_right:
            # Refine the corner locations
            corners_left_refined = cv2.cornerSubPix(left_gray, corners_left, (11,11), (-1,-1), criteria)
            corners_right_refined = cv2.cornerSubPix(right_gray, corners_right, (11,11), (-1,-1), criteria)
            
            # Store results
            all_objpoints.append(base_objp)
            left_imgpoints_final.append(corners_left_refined)
            right_imgpoints_final.append(corners_right_refined)
            
            # Optional display
            if display:
                cv2.drawChessboardCorners(left_img, (cols, rows), corners_left_refined, True)
                cv2.drawChessboardCorners(right_img, (cols, rows), corners_right_refined, True)
                combined = cv2.hconcat([left_img, right_img])
                cv2.imshow("Stereo Pair Corners (Left | Right)", combined)
                cv2.waitKey(200)

    if display:
        cv2.destroyAllWindows()

    # Check if we found enough pairs
    if len(all_objpoints) < 2:
        raise ValueError("Not enough valid stereo pairs were found. Check your images or chessboard size.")

    # 3) Calibrate the left camera alone
    cameraMatrix1, distCoeffs1, rvecs1, tvecs1 = calibrate_single_camera(
        all_objpoints, left_imgpoints_final, image_size
    )

    # 4) Calibrate the right camera alone
    cameraMatrix2, distCoeffs2, rvecs2, tvecs2 = calibrate_single_camera(
        all_objpoints, right_imgpoints_final, image_size
    )

    print("Single camera calibration done.")
    print("Left camera matrix:\n", cameraMatrix1)
    print("Left distortion:\n", distCoeffs1)
    print("Right camera matrix:\n", cameraMatrix2)
    print("Right distortion:\n", distCoeffs2)

    # 5) Perform stereo calibration
    R, T, E, F = stereo_calibrate(
        all_objpoints,
        left_imgpoints_final,
        right_imgpoints_final,
        cameraMatrix1,
        distCoeffs1,
        cameraMatrix2,
        distCoeffs2,
        image_size
    )

    print("\nStereo Calibration Results:")
    print("Rotation between cameras (R):\n", R)
    print("Translation between cameras (T):\n", T)
    print("Essential matrix (E):\n", E)
    print("Fundamental matrix (F):\n", F)

    # 6) (Optional) Stereo Rectification
    R1, R2, P1, P2, Q = stereo_rectify(
        cameraMatrix1, distCoeffs1,
        cameraMatrix2, distCoeffs2,
        image_size,
        R, T
    )

    print("\nRectification matrices:")
    print("R1:\n", R1)
    print("R2:\n", R2)
    print("P1:\n", P1)
    print("P2:\n", P2)
    print("Q:\n", Q)

    # You can now use R1, R2, P1, P2, and Q to warp stereo images for depth estimation
    # via cv2.initUndistortRectifyMap() and cv2.remap().

        # 7) Process real-time stereo images for depth estimation
    process_stereo_images(
        R1, R2, P1, P2, Q,
        cameraMatrix1, distCoeffs1,
        cameraMatrix2, distCoeffs2,
        image_size
    )


if __name__ == "__main__":
    """
    Usage:
    - Two folders: "left" and "right", each containing synchronized images of the 11x8 checkerboard.
    - square_size = 25.0 means each square is 25 mm wide/tall.
    """
    main(
        left_images_glob="stereo_camera/calibration_images/left/*.jpg",
        right_images_glob="stereo_camera/calibration_images/right/*.jpg",
        chessboard_size=(10, 7),
        square_size=25.0,  # 25 mm per checkerboard square
        display=False
    )
