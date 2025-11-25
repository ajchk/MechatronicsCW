"""
Mechatronics_Master_fixed.py
Clean and stable version of the Mechatronics master script.
Features:
- Non-blocking UDP listener for 'ENDSTOP'
- Capture snapshot on ENDSTOP and save ArUco ID/x data as NumPy .npz
- Visual display with live ArUco overlay
"""
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import socket
import json
import datetime

UDP_IP_SEND = "138.38.228.99"
UDP_IP_RECEIVE = "172.26.109.96"
UDP_PORT = 25000
MARKER_SIZE_MM = 40
PROCESSING_PERIOD = 0.25


def save_id_x_pairs_npz(ids, tvecs, timestamp):
    id_list = ids.flatten().astype(np.int32)
    x_arr = np.array([float(np.squeeze(tv)[0]) for tv in tvecs], dtype=np.float64)
    npz_filename = f"aruco_id_x_{timestamp}.npz"
    json_filename = f"aruco_id_x_{timestamp}.json"
    np.savez_compressed(npz_filename, ids=id_list, x=x_arr)
    summary = [{'id': int(i), 'x': float(x)} for i, x in zip(id_list.tolist(), x_arr.tolist())]
    with open(json_filename, 'w') as jf:
        json.dump(summary, jf, indent=2)
    print(f"Saved {npz_filename} and {json_filename}")


def open_receive_socket(ip, port, non_blocking=True):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((ip, port))
    except OSError:
        sock.bind(('0.0.0.0', port))
    if non_blocking:
        sock.setblocking(False)
    print("Listening on", sock.getsockname())
    return sock


def main():
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv = open_receive_socket(UDP_IP_RECEIVE, UDP_PORT, non_blocking=True)

    try:
        camera_calibration = np.load('Calibration.npz')
        CM = camera_calibration['CM']
        dist_coef = camera_calibration['dist_coef']
    except Exception:
        print('Calibration load failed; continuing without pose estimation')
        CM = None
        dist_coef = None

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

    start_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('No camera frame; exiting')
                break
            corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                if CM is not None and dist_coef is not None:
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_MM, CM, dist_coef)
                    for rvec, tvec in zip(rvecs, tvecs):
                        cv2.drawFrameAxes(frame, CM, dist_coef, rvec, tvec, 100)

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Frame', frame)

            # Non-blocking UDP recv
            try:
                data, addr = sock_recv.recvfrom(1024)
                msg = data.decode('utf-8').strip()
                print('UDP msg:', msg)
                if msg == 'ENDSTOP':
                    # Take snapshot and save id/x
                    ret_s, snap = cap.read()
                    if ret_s:
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        cv2.imwrite(f'photo_{timestamp}.png', snap)
                        corners2, ids2, _ = aruco.detectMarkers(snap, aruco_dict, parameters=parameters)
                        if ids2 is not None and CM is not None and dist_coef is not None:
                            rvecs2, tvecs2, _ = aruco.estimatePoseSingleMarkers(corners2, MARKER_SIZE_MM, CM, dist_coef)
                            save_id_x_pairs_npz(ids2, tvecs2, timestamp)
                        else:
                            print('No markers or calibration; saving image only')
            except BlockingIOError:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed_time = time.time() - start_time
            fps = 1.0 / (elapsed_time if elapsed_time > 0 else 1e-6)
            if elapsed_time < PROCESSING_PERIOD:
                time.sleep(PROCESSING_PERIOD - elapsed_time)
            start_time = time.time()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock_recv.close()
        sock_send.close()


if __name__ == '__main__':
    main()
