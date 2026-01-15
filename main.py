import cv2
import numpy as np
import math

# ================================
MARKER_SIZE = 0.08      # kích thước marker (m) – đúng với marker bạn in
CUBE_SIZE   = 0.12      # cạnh khối lập phương 12cm
# ================================

# Camera intrinsics (tạm, nên calibrate sau)
cameraMatrix = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)

distCoeffs = np.zeros((5,1))

# ===== marker offset từ tâm =====
marker_offsets = {
    0: np.array([0, 0,  CUBE_SIZE/2]),   # trước
    1: np.array([0, 0, -CUBE_SIZE/2]),   # sau
    2: np.array([-CUBE_SIZE/2, 0, 0]),   # trái
    3: np.array([ CUBE_SIZE/2, 0, 0]),   # phải
    4: np.array([0, -CUBE_SIZE/2, 0]),   # trên
    5: np.array([0,  CUBE_SIZE/2, 0])    # dưới
}

# ===== Pose ban đầu =====
init_center = None
init_R = None

# ===== rotation → Euler =====
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.degrees([x, y, z])   # Roll, Pitch, Yaw

# ===== ArUco =====
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

print(">> Đưa khối vào camera – frame đầu tiên sẽ được dùng làm gốc (0,0,0)")

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

            # vẽ trục marker
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.03)

            R, _ = cv2.Rodrigues(rvec)
            offset = marker_offsets[marker_id].reshape(3,1)

            # tâm khối = vị trí marker − R * offset
            center_pos = tvec - R @ offset

            centers.append(center_pos)
            rotations.append(R)

    # ===== Tính DELTA pose =====
    if len(centers) > 0:
        center_pos = np.mean(centers, axis=0)
        R = rotations[0]

        # Lưu pose ban đầu
        if init_center is None:
            init_center = center_pos.copy()
            init_R = R.copy()
            print(">> Initial pose captured")

        # Δ vị trí
        delta_pos = center_pos - init_center

        # Δ xoay
        delta_R = R @ init_R.T

        roll, pitch, yaw = rotationMatrixToEulerAngles(delta_R)
        dx, dy, dz = delta_pos.flatten()

        cv2.putText(frame, f"DELTA X:{dx:.3f} Y:{dy:.3f} Z:{dz:.3f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(frame, f"Roll:{roll:.1f} Pitch:{pitch:.1f} Yaw:{yaw:.1f}",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Digital Twin Tracker", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
