import cv2
import cv2.aruco as aruco
import numpy as np
import time
import socket






                    #------CAMERA CALIBRATION------
# Load the camera calibration values
camera_calibration = np.load('Calibration.npz')
CM=camera_calibration['CM'] #camera matrix
dist_coef=camera_calibration['dist_coef']# distortion coefficients from the camera

# Define the ArUco dictionary and parameters
marker_size = 83
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Define a processing rate
processing_period = 0.25

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set the starting time
start_time = time.time()
fps = 0

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

# Set the starting time
start_time = time.time()
fps = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Detect markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    # If markers are detected
    if ids is not None:
        # Draw detected markers
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of each marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, CM, dist_coef)

        for rvec, tvec in zip(rvecs, tvecs):
            # Draw axis for each marker
            frame = cv2.drawFrameAxes(frame, CM, dist_coef, rvec, tvec, 100)

    # Add the frame rate to the image
    cv2.putText(frame, f"CAMERA FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"PROCESSING FPS: {1/processing_period:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Non-blocking UDP receive

  
            # --- Take photo and save ArUco IDs ---
ret, frame = cap.read()
if ret:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_filename = f"image_at_x0.png"
                cv2.imwrite(img_filename, frame)
                print(f"Photo taken and saved as {img_filename}")
                corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
                if ids is not None:
                    print(f"Detected ArUco IDs: {ids.flatten().tolist()}")
                    with open(f"aruco_ids_{timestamp}.txt", "w") as f:
                        f.write(",".join(map(str, ids.flatten().tolist())))
                else:
                    print("No ArUco markers detected in the photo.")
else:
    print("No ArUco markers detected in the photo.")
        
    