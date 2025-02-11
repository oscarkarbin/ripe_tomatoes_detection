import cv2
import threading

# Open both cameras
capL = cv2.VideoCapture(1)  # Left camera
capR = cv2.VideoCapture(2)  # Right camera


def set_camera_settings(cap):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Disable auto-exposure (Windows)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

set_camera_settings(capL)
set_camera_settings(capR)

frameL, frameR = None, None
lock = threading.Lock()

def capture_left():
    global frameL
    while True:
        ret, frame = capL.read()
        if ret:
            with lock:
                frameL = frame

def capture_right():
    global frameR
    while True:
        ret, frame = capR.read()
        if ret:
            with lock:
                frameR = frame

# Start threads
threadL = threading.Thread(target=capture_left, daemon=True)
threadR = threading.Thread(target=capture_right, daemon=True)
threadL.start()
threadR.start()

frame_id = 0

while True:
    with lock:
        if frameL is not None and frameR is not None:
            cv2.imshow("Left", frameL)
            cv2.imshow("Right", frameR)

    key = cv2.waitKey(1)
    
    # Press 's' to save both images simultaneously
    if key & 0xFF == ord('s'):
        with lock:
            cv2.imwrite(f"left_{frame_id}.jpg", frameL)
            cv2.imwrite(f"right_{frame_id}.jpg", frameR)
            print(f"Saved left_{frame_id}.jpg and right_{frame_id}.jpg")
            frame_id += 1

    # Press 'q' to quit
    if key & 0xFF == ord('q'):
        break

# Cleanup
capL.release()
capR.release()
cv2.destroyAllWindows()
