import cv2
import numpy as np
import math

MARKER_SIZE = 0.05  # 5cm

cameraMatrix = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)

distCoeffs = np.zeros((5,1))

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

    return np.degrees([x, y, z])

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, cameraMatrix, distCoeffs
        )

        for i in range(len(ids)):
            rvec = rvecs[i]
            tvec = tvecs[i]

            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.03)

            R, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = rotationMatrixToEulerAngles(R)

            x, y, z = tvec[0]

            cv2.putText(frame, f"ID {ids[i][0]}", (10, 30+i*60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.putText(frame, f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}",
                        (10, 50+i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.putText(frame, f"Roll:{roll:.1f} Pitch:{pitch:.1f} Yaw:{yaw:.1f}",
                        (10, 70+i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Aruco Pose", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
