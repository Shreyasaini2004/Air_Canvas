# Air Canvas ✋🎨

Draw in the air using color-based object tracking with OpenCV. Move a
colored marker (e.g., a colored bottle cap) in front of your webcam to
paint on a virtual canvas — no touchscreen or stylus required.

## How it works

1. Captures live webcam video and converts each frame to HSV color space.
2. Uses adjustable trackbars to filter for a specific color range (HSV
   thresholding), producing a binary mask of the tracked object.
3. Cleans up the mask with morphological operations (erosion, opening,
   dilation) to reduce noise.
4. Finds the largest contour in the mask and tracks its centroid as the
   "pen tip."
5. Records the pen tip's path per color and renders it as connected line
   segments on both the live feed and a separate white canvas.
6. On-screen buttons let you switch between blue / green / red / yellow,
   or clear the canvas.

## Setup

```bash
pip install -r requirements.txt
python air_canvas.py
```

## Controls

- Move your colored marker in front of the camera to draw.
- Hover over a color swatch (top of the window) to switch colors.
- Hover over **CLEAR** to reset the canvas.
- Use the **Color Detector** trackbar window to tune HSV thresholds for
  your specific marker color and lighting conditions.
- Press **q** to quit.

## Tech

Python, OpenCV, NumPy
