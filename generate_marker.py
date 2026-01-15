import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# =========================
MARKER_SIZE_CM = 8     # kích thước thật
MARKER_IDS = [0,1,2,3,4,5]
DPI = 300             # độ phân giải in
# =========================

aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# đổi cm → pixel
marker_size_px = int(MARKER_SIZE_CM * DPI / 2.54)

# tạo canvas A4
width, height = A4
c = canvas.Canvas("aruco_markers_A4.pdf", pagesize=A4)

x = 50
y = height - 50

for marker_id in MARKER_IDS:
    # tạo marker
    marker = cv2.aruco.generateImageMarker(aruco, marker_id, marker_size_px)

    # lưu tạm
    filename = f"temp_{marker_id}.png"
    cv2.imwrite(filename, marker)

    # xuống dòng khi hết ngang
    if x + MARKER_SIZE_CM*28 > width:
        x = 50
        y -= MARKER_SIZE_CM*28 + 50

    # vẽ vào PDF
    c.drawImage(filename, x, y - MARKER_SIZE_CM*28,
                width=MARKER_SIZE_CM*28,
                height=MARKER_SIZE_CM*28)

    # vẽ ID
    c.drawString(x, y - MARKER_SIZE_CM*28 - 15, f"ID {marker_id}")

    x += MARKER_SIZE_CM*28 + 40

c.save()

print("Đã tạo file aruco_markers_A4.pdf — in ở chế độ 100% scale")
