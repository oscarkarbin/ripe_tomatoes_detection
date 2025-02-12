import cv2
import numpy as np
from ultralytics import YOLO

# Load calibration data (Q matrix for disparity-depth conversion)
calib_data = np.load("stereo_calibration_data.npz")
Q = calib_data["Q"]
mtxL, distL = calib_data["mtxL"], calib_data["distL"]
mtxR, distR = calib_data["mtxR"], calib_data["distR"]
mapL1, mapL2, mapR1, mapR2 = calib_data["mapL1"], calib_data["mapL2"], calib_data["mapR1"], calib_data["mapR2"]

#model path
model_path = "yolo_models/yolo11s.pt"

#Frame Width and height
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Load YOLOv11s model (Modify path as needed)
model = YOLO(model_path)

# Open stereo cameras
capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(2)

capL.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Compute disparity map
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
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

while True:
    # Capture frames
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Error: Could not read frames.")
        break

    # Convert to grayscale for disparity computation
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    rectifiedL = cv2.remap(grayL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(grayR, mapR1, mapR2, cv2.INTER_LINEAR)

    disparity = stereo.compute(rectifiedL, rectifiedR).astype(np.float32) / 16.0

    disparity_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_vis = np.uint8(disparity_vis)

    # Run YOLOv11s on the left frame
    results = model.predict(source=frameL)

    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    classes = results[0].boxes.cls.cpu().numpy()  # Class IDs


    for (box, conf, cls) in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = map(int, box)  # Convert bbox coordinates to int

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Get disparity at bounding box center
        disp_value = disparity[center_y, center_x]
        if disp_value <= 0:
            continue  # Ignore invalid disparities

        # Convert disparity to 3D coordinates (X, Y, Z)
        point_3D = cv2.perspectiveTransform(np.array([[[center_x, center_y, disp_value]]], dtype=np.float32), Q)
        distance = point_3D[0][0][2]  # Extract Z value (depth in meters)

        # Draw detection and distance
        cv2.rectangle(frameL, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frameL, f"{distance:.2f}m", (center_x, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Display results

    for y in range(0, FRAME_HEIGHT, 50):  # Draw horizontal lines every 50 pixels
        cv2.line(rectifiedL, (0, y), (FRAME_WIDTH, y), (255, 0, 0), 1)
        cv2.line(rectifiedR, (0, y), (FRAME_WIDTH, y), (255, 0, 0), 1)

    cv2.imshow("Left Rectified", rectifiedL)
    cv2.imshow("Right Rectified", rectifiedR)

    cv2.imshow("Left Camera with Detections", frameL)
    cv2.imshow("Disparity Map", disparity_vis)  # Normalize for visibility

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
