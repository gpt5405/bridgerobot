#!/usr/bin/env python3
import cv2
import numpy as np
import time

# --------- TUNE ME (if needed) ----------
TARGET_RGB = (213, 59, 119)   # R, G, B as requested
H_TOL = 10                    # ± Hue tolerance (0-179 in OpenCV); start with 10
S_TOL = 60                    # ± Saturation tolerance (0-255)
V_TOL = 60                    # ± Value/Brightness tolerance (0-255)
MIN_AREA = 1200               # ignore tiny specks (pixels)
RESIZE_WIDTH = 640            # downscale for speed; set to 0 to keep native
CAM_INDEX = 0                 # 0 for /dev/video0. Change if you have multiple cams.
# ----------------------------------------

def compute_hsv_bounds(rgb, h_tol=10, s_tol=60, v_tol=60):
    """
    Convert an RGB target to HSV and generate lower/upper bounds with tolerance.
    Handles hue wrap-around in OpenCV's 0..179 hue space.
    Returns one or two (lower, upper) tuples; if wrap occurs, returns two ranges.
    """
    r, g, b = rgb
    # OpenCV expects BGR
    bgr = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Bound S/V within [0, 255]
    s_low, s_high = max(0, s - s_tol), min(255, s + s_tol)
    v_low, v_high = max(0, v - v_tol), min(255, v + v_tol)

    # Hue wrap handling in 0..179
    h_low = h - h_tol
    h_high = h + h_tol
    if h_low < 0:
        # two segments: [0..h_high] and [180+h_low .. 179]
        range1 = (np.array([0, s_low, v_low], dtype=np.uint8),
                  np.array([h_high, s_high, v_high], dtype=np.uint8))
        range2 = (np.array([180 + h_low, s_low, v_low], dtype=np.uint8),
                  np.array([179, s_high, v_high], dtype=np.uint8))
        return [range1, range2]
    elif h_high > 179:
        # two segments: [h_low..179] and [0..(h_high-180)]
        range1 = (np.array([h_low, s_low, v_low], dtype=np.uint8),
                  np.array([179, s_high, v_high], dtype=np.uint8))
        range2 = (np.array([0, s_low, v_low], dtype=np.uint8),
                  np.array([h_high - 180, s_high, v_high], dtype=np.uint8))
        return [range1, range2]
    else:
        # single continuous range
        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
        return [(lower, upper)]

def main():
    hsv_ranges = compute_hsv_bounds(TARGET_RGB, H_TOL, S_TOL, V_TOL)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    # Prefer a smaller frame for speed
    if RESIZE_WIDTH > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(RESIZE_WIDTH * 9 / 16))  # assume ~16:9

    # Try to keep latency low
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    last_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed. Check CAM_INDEX or camera connection.")
            break

        # Optional resize for speed (preserve aspect ratio)
        if RESIZE_WIDTH > 0:
            h, w = frame.shape[:2]
            if w != RESIZE_WIDTH:
                scale = RESIZE_WIDTH / float(w)
                frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # Convert to HSV once
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold with one or two ranges (to handle hue wrap)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in hsv_ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        # Clean up noise (fast morphology)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours of the detected regions
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw boxes + label
        for c in cnts:
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 255, 60), 2)
            cv2.putText(frame, "RUST", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 60), 2, cv2.LINE_AA)

        # FPS (optional overlay)
        now = time.time()
        dt = now - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        last_t = now
        cv2.putText(frame, f"{fps:0.1f} FPS", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)

        cv2.imshow("RUST color detection", frame)
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()