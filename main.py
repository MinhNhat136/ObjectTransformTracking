import cv2
import numpy as np
import math
import socket
import json
from entity_socket import SocketData
from socket_helper import SocketHelper

# ================================
MARKER_SIZE = 0.08      # kích thước marker (m) – đúng với marker bạn in
CUBE_SIZE   = 0.12      # cạnh khối lập phương 12cm
# ================================

# Camera intrinsics (tạm, nên calibrate)
cameraMatrix = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)

distCoeffs = np.zeros((5,1))

# ===== rotation helpers =====
def rot_x(a):
    return np.array([
        [1,0,0],
        [0,np.cos(a),-np.sin(a)],
        [0,np.sin(a), np.cos(a)]
    ])

def rot_y(a):
    return np.array([
        [ np.cos(a),0,np.sin(a)],
        [0,1,0],
        [-np.sin(a),0,np.cos(a)]
    ])

# ===== offset từ tâm =====
marker_offsets = {
    0: np.array([0, 0,  CUBE_SIZE/2]),   # trước
    1: np.array([0, 0, -CUBE_SIZE/2]),   # sau
    2: np.array([-CUBE_SIZE/2, 0, 0]),   # trái
    3: np.array([ CUBE_SIZE/2, 0, 0]),   # phải
    4: np.array([0, -CUBE_SIZE/2, 0]),   # trên
    5: np.array([0,  CUBE_SIZE/2, 0])    # dưới
}

# ===== orientation của marker so với khối =====
R_cube_marker = {
    0: np.eye(3),              # trước
    1: rot_y(np.pi),           # sau
    2: rot_y(np.pi/2),         # trái
    3: rot_y(-np.pi/2),        # phải
    4: rot_x(-np.pi/2),        # trên
    5: rot_x(np.pi/2)          # dưới
}

# ===== pose gốc =====
init_center = None
init_R = None

# ===== rotation → Euler =====
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.degrees([x,y,z])

def send_transform(sock, transform):
    sock_data = SocketData()

    SocketHelper.write_payload(sock_data, transform)

    sock.sendall(sock_data.get_binary())

# ===== ArUco =====
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

print(">> Đưa khối vào camera – frame đầu tiên sẽ là pose gốc")

# ===== Socket Connection =====
HOST = '127.0.0.1'
PORT = 8052
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((HOST, PORT))
    print(f">> Connected to {HOST}:{PORT}")
except Exception as e:
    print(f"!! Failed to connect to {HOST}:{PORT}: {e}")
    sock = None

    sock = None

prev_time = 0
last_send_time = 0
SEND_INTERVAL = 1.0 / 30  # 30 FPS

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)

    centers = []
    rotations = []

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, cameraMatrix, distCoeffs
        )

        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            if marker_id not in marker_offsets:
                continue

            rvec = rvecs[i]
            tvec = tvecs[i].reshape(3,1)

            R_cm, _ = cv2.Rodrigues(rvec)        # camera → marker
            R_mc = R_cube_marker[marker_id]     # cube → marker

            # camera → cube
            R_cc = R_cm @ R_mc.T

            offset = marker_offsets[marker_id].reshape(3,1)

            # center = marker − R_cc * offset
            center_pos = tvec - R_cc @ offset

            centers.append(center_pos)
            rotations.append(R_cc)

            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.03)

    if len(centers) > 0:
        center_pos = np.mean(centers, axis=0)
        R = rotations[0]

        if init_center is None:
            init_center = center_pos.copy()
            init_R = R.copy()
            print(">> Initial pose captured")

        # Δ position
        delta_pos = center_pos - init_center

        # Δ rotation
        delta_R = R @ init_R.T

        roll, pitch, yaw = rotationMatrixToEulerAngles(delta_R)
        dx,dy,dz = delta_pos.flatten()

        cv2.putText(frame, f"DELTA X:{dx:.3f} Y:{dy:.3f} Z:{dz:.3f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Roll:{roll:.1f} Pitch:{pitch:.1f} Yaw:{yaw:.1f}",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)


        # ===== Send Data via Socket =====
        if sock and (curr_time - last_send_time >= SEND_INTERVAL):
            try:
                # Prepare data list (rounded to 1 decimal place)
                data = [round(float(dx), 1), round(float(dy), 1), round(float(dz), 1), 
                        round(float(roll), 1), round(float(pitch), 1), round(float(yaw), 1)]
                json_data = json.dumps(data)
                
                # Pack data
                socket_data = SocketData()
                SocketHelper.write_payload(socket_data, json_data)
                
                # Send binary
                sock.sendall(socket_data.get_binary())
                last_send_time = curr_time
            except Exception as e:
                print(f"!! Socket send error: {e}")
                sock = None

    cv2.imshow("Digital Twin Rigid Body", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
if sock:
    sock.close()
cv2.destroyAllWindows()
