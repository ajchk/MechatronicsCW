import cv2
import numpy as np

# ---------- Config ----------
image_path = "image_at_xX.png"      # your input photo
calib_path = "Calibration.npz"     # your calibration file
marker_length_mm = 83         # change to your actual marker size in mm

# ---------- Load calibration ----------
camera_calibration = np.load('Calibration.npz')
CM=camera_calibration['CM'] #camera matrix
dist_coef=camera_calibration['dist_coef']# distortion coefficients from the camera

# ---------- Load image ----------
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

# ---------- Detect ArUco markers ----------
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # change if needed
parameters = aruco.DetectorParameters()

corners, ids, rejected = aruco.detectMarkers(img, dictionary, parameters=parameters)

if ids is None or len(ids) == 0:
    print("No ArUco markers detected.")
else:
    # Draw marker borders
    aruco.drawDetectedMarkers(img, corners, ids)

    # Estimate pose of each marker
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
        corners, marker_length_mm, CM, dist_coef
    )

    # Print x coordinates (in mm)
    x_coords_mm = tvecs[:, 0, 0]  # shape: (N,)
    print("X coordinates of markers in mm:")
    for marker_id, x in zip(ids.flatten(), x_coords_mm):
        print(f"Marker {marker_id}: X = {x:.2f} mm")

    # Overlay XYZ on the image
    for i, marker_id in enumerate(ids.flatten()):
        rvec = rvecs[i]
        tvec = tvecs[i, 0]  # (X, Y, Z) in mm

        # Draw axis for visualization (optional)
        """ aruco.drawAxis(img, CM, dist_coef, rvec, tvec, marker_length_mm * 0.5) """

        # Position for text (use first corner of marker)
        corner = corners[i][0][0]  # top-left corner of marker
        x_img, y_img = int(corner[0]), int(corner[1])

        text = f"X:{tvec[0]:.1f} Y:{tvec[1]:.1f} Z:{tvec[2]:.1f} mm"
        cv2.putText(
            img, text, (x_img, y_img - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
        )

# ---------- Save / show result ----------
output_path = "aruco_with_xyz.png"
cv2.imwrite(output_path, img)
print(f"Annotated image saved to {output_path}")
