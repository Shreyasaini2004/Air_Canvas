"""
Air Canvas — draw in the air using color-based hand/marker tracking.

Tracks a colored object (e.g., a colored cap/marker) via webcam using HSV
color filtering, then lets the user "paint" on a virtual canvas by moving
that object in front of the camera.
"""

import numpy as np          # numerical computation - matrix, arrays
import cv2                  # computer vision / webcam handling
from collections import deque   # used to store drawing points efficiently


# ---------------------------------------------------------------------------
# PART 1 — Trackbar setup (for live HSV color-range tuning)
# ---------------------------------------------------------------------------

def setValues(x):
    """Callback function — required by createTrackbar, does nothing itself.
    Called automatically whenever a trackbar value changes."""
    pass


cv2.namedWindow("Color Detector")
# HSV (Hue, Saturation, Value) ranges: H -> 0-180, S -> 0-255, V -> 0-255
# 6 trackbars: upper/lower hue, saturation, and value
# Default values below are pre-tuned to detect a blue-colored object.
cv2.createTrackbar("Upper Hue", "Color Detector", 153, 180, setValues)
cv2.createTrackbar("Upper Saturation", "Color Detector", 255, 255, setValues)
cv2.createTrackbar("Upper Value", "Color Detector", 255, 255, setValues)
cv2.createTrackbar("Lower Hue", "Color Detector", 64, 180, setValues)
cv2.createTrackbar("Lower Saturation", "Color Detector", 72, 255, setValues)
cv2.createTrackbar("Lower Value", "Color Detector", 49, 255, setValues)


# ---------------------------------------------------------------------------
# PART 2 — Color buffers and drawing setup
# ---------------------------------------------------------------------------

# Each color gets its own list of deques — one deque per continuous stroke.
# maxlen=512 caps memory usage per stroke.
blue_points = [deque(maxlen=512)]
green_points = [deque(maxlen=512)]
red_points = [deque(maxlen=512)]
yellow_points = [deque(maxlen=512)]

# Pointers to the current (active) stroke/deque for each color.
# A new deque is appended whenever a new stroke begins for that color.
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Kernel for morphological operations (noise cleanup on the color mask).
# 5x5 matrix of 1s, dtype uint8 (values 0-255).
# Helps close small gaps in the detected color blob and remove stray noise.
kernel = np.ones((5, 5), np.uint8)


# ---------------------------------------------------------------------------
# PART 3 — Canvas setup
# ---------------------------------------------------------------------------

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # B, G, R, Y
colourIndex = 0  # currently selected drawing color

# White canvas: 471x636 px, 3 channels (RGB). np.full(..., 255) -> white
# (np.zeros would give a black canvas by default).
paintWindow = np.full((471, 636, 3), 255, dtype=np.uint8)

# Draw the UI buttons (clear + 4 color swatches) onto the canvas.
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (510, 1), (590, 65), colors[3], -1)
# -1 fill thickness = solid filled rectangle (button). 2 = outline only (clear button).

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# PART 4 — Main loop: capture frames, detect color, draw
# ---------------------------------------------------------------------------

cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    # ret -> whether the frame was read successfully
    # frame -> the actual captured image

    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror the feed for a more natural feel
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read current HSV threshold values from the trackbars.
    u_hue = cv2.getTrackbarPos("Upper Hue", "Color Detector")
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color Detector")
    u_sat = cv2.getTrackbarPos("Upper Saturation", "Color Detector")
    l_sat = cv2.getTrackbarPos("Lower Saturation", "Color Detector")
    u_val = cv2.getTrackbarPos("Upper Value", "Color Detector")
    l_val = cv2.getTrackbarPos("Lower Value", "Color Detector")

    Upper_hsv = np.array([u_hue, u_sat, u_val])
    Lower_hsv = np.array([l_hue, l_sat, l_val])

    # Draw the same UI buttons on the live webcam feed.
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), -1)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], -1)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), colors[1], -1)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), colors[2], -1)
    frame = cv2.rectangle(frame, (510, 1), (590, 65), colors[3], -1)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Binary mask: white where pixels fall inside the selected HSV range.
    mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)

    # Clean up the mask: remove small noise, then restore/connect the blob.
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours (outer boundaries) of the detected color blob(s).
    # [-2:] handles both OpenCV 3.x and 4.x return signatures.
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2:]
    center = None  # center point of the tracked color blob, if found

    if len(cnts) > 0:
        # Track only the largest detected contour (assumed to be the marker).
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)

        if radius > 20:
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       colors[colourIndex], 2)

            # Contour moments -> compute the centroid (x, y) of the blob.
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            # PART 5 — Button handling
            if center[1] <= 65:
                if 40 <= center[0] <= 140:  # Clear button
                    blue_points = [deque(maxlen=512)]
                    green_points = [deque(maxlen=512)]
                    red_points = [deque(maxlen=512)]
                    yellow_points = [deque(maxlen=512)]

                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0

                    paintWindow[67:, :, :] = 255  # reset canvas to white
                elif 160 <= center[0] <= 255:
                    colourIndex = 0  # blue
                elif 275 <= center[0] <= 370:
                    colourIndex = 1  # green
                elif 390 <= center[0] <= 485:
                    colourIndex = 2  # red
                elif 510 <= center[0] <= 590:
                    colourIndex = 3  # yellow
            else:
                # Not over a button -> add this point to the active stroke.
                if colourIndex == 0:
                    blue_points[blue_index].appendleft(center)
                elif colourIndex == 1:
                    green_points[green_index].appendleft(center)
                elif colourIndex == 2:
                    red_points[red_index].appendleft(center)
                elif colourIndex == 3:
                    yellow_points[yellow_index].appendleft(center)
    else:
        # No color detected this frame -> start a new stroke for each color
        # (prevents lines being drawn across gaps when tracking is lost).
        blue_points.append(deque(maxlen=512))
        blue_index += 1

        green_points.append(deque(maxlen=512))
        green_index += 1

        red_points.append(deque(maxlen=512))
        red_index += 1

        yellow_points.append(deque(maxlen=512))
        yellow_index += 1

    # PART 6 — Render all stored strokes, for every color, onto both windows.
    points = [blue_points, green_points, red_points, yellow_points]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k],
                          colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k],
                          colors[i], 2)

    cv2.imshow("Live Drawing", frame)
    cv2.imshow("Paint", paintWindow)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()