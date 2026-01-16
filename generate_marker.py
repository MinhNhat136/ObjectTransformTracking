import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

MARKER_SIZE_CM = 12
MARKER_IDS = [0,1,2,3,4,5]
DPI = 300

CM_TO_PT = 72 / 2.54

aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

marker_size_px = int(MARKER_SIZE_CM * DPI / 2.54)
marker_size_pt = MARKER_SIZE_CM * CM_TO_PT

width, height = A4
c = canvas.Canvas("aruco_markers_A4.pdf", pagesize=A4)

margin = 40
x = margin
y = height - margin

for marker_id in MARKER_IDS:
    marker = cv2.aruco.generateImageMarker(aruco, marker_id, marker_size_px)

    filename = f"temp_{marker_id}.png"
    cv2.imwrite(filename, marker)

    if x + marker_size_pt > width - margin:
        x = margin
        y -= marker_size_pt + 40

    c.drawImage(
        filename,
        x,
        y - marker_size_pt,
        width=marker_size_pt,
        height=marker_size_pt
    )

    c.drawString(x, y - marker_size_pt - 15, f"ID {marker_id}")

    x += marker_size_pt + 40

c.save()
print("OK — marker 12cm, in 100% scale")
