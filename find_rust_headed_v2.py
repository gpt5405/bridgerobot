#!/usr/bin/env python3
"""
Rust color detector with *tight* RGB matching and servo control (Raspberry Pi).

- Target color: RGB(213, 59, 119) with ±5% relative tolerance on each channel.
- When detected:
    1) SG90 -> 180°, wait 1s
    2) MS24 -> 15°, hold 1s, then -> 0°
    3) wait 1s
    4) SG90 -> 0°
- Live OpenCV preview with bounding boxes and status text (disable with --no-gui).
- Debounce so the sequence doesn't fire continuously when the color stays in view.

Dependencies:
  pip install --no-cache-dir opencv-python adafruit-blinka adafruit-circuitpython-servokit
  sudo apt update && sudo apt install -y python3-smbus i2c-tools
  sudo raspi-config  # Interface Options -> I2C -> Enable

Run:
  python3 find_rust_headed.py
  # optional args:
  # python3 find_rust_headed.py --cam 0 --area-min 400 --debounce-sec 2.0

Notes:
- If you want an even tighter window, lower --tol 0.05 to e.g. 0.03 (±3%).
- This matches in RGB *after* converting from BGR (OpenCV capture is BGR).
"""

import sys
import time
import argparse
from typing import Tuple, List

import cv2
import numpy as np

# -----------------------------
# Servo / actuator configuration
# -----------------------------

# PCA9685 channels (adjust to your wiring)
CH_SG90 = 0   # Micro Servo SG90
CH_MS24 = 1   # Digital Servo MS24 20kg

# Angle limits (sane defaults; tweak if your linkage requires different end-stops)
SG90_MIN_ANGLE = 0
SG90_MAX_ANGLE = 180

MS24_MIN_ANGLE = 0
MS24_MAX_ANGLE = 180

# PCA9685 parameters
PCA9685_I2C_ADDR = 0x40
PCA9685_FREQ_HZ = 50

# -----------------------------
# Detection configuration
# -----------------------------

TARGET_RGB_DEFAULT = (213, 59, 119)  # R, G, B
REL_TOL_DEFAULT = 0.05               # ±5% of each channel's own value
AREA_MIN_DEFAULT = 400               # min contour area to count as detection
DEBOUNCE_SEC_DEFAULT = 2.0           # minimum seconds between fire sequences
CAM_INDEX_DEFAULT = 0

# Morphology to clean noise
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# -----------------------------
# Servo helpers
# -----------------------------
def init_servos():
    """
    Initialize PCA9685 and return a ServoKit instance.
    """
    try:
        from adafruit_servokit import ServoKit
        kit = ServoKit(channels=16, address=PCA9685_I2C_ADDR)
        # Set frequency
        try:
            kit.frequency = PCA9685_FREQ_HZ  # some versions support this attribute
        except Exception:
            pass

        # Optional: set pulse width range per servo if needed
        # kit.servo[CH_SG90].set_pulse_width_range(500, 2500)
        # kit.servo[CH_MS24].set_pulse_width_range(500, 2500)

        # Park servos at 0 on start
        safe_servo_angle(kit, CH_SG90, 0, SG90_MIN_ANGLE, SG90_MAX_ANGLE)
        safe_servo_angle(kit, CH_MS24, 0, MS24_MIN_ANGLE, MS24_MAX_ANGLE)
        return kit
    except Exception as e:
        print(f"[Servo] Init failed: {e}", file=sys.stderr)
        raise

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def safe_servo_angle(kit, ch, angle, amin, amax):
    try:
        kit.servo[ch].angle = float(clamp(angle, amin, amax))
    except Exception as e:
        print(f"[Servo] Set angle failed ch={ch}: {e}", file=sys.stderr)

def smooth_move_servo(kit, ch, target, steps=8, delay=0.03, amin=0, amax=180):
    """Small stepped motion for less jerk."""
    try:
        current = kit.servo[ch].angle
    except Exception:
        current = None
    if current is None:
        safe_servo_angle(kit, ch, target, amin, amax)
        time.sleep(delay)
        return
    target = clamp(target, amin, amax)
    delta = target - current
    if steps < 1:
        steps = 1
    for i in range(1, steps + 1):
        a = current + (delta * i / steps)
        safe_servo_angle(kit, ch, a, amin, amax)
        time.sleep(delay)

def run_sequence(kit):
    """
    Your specified sequence:
      SG90 -> 180°, wait 1s
      MS24 -> 15°, hold 1s, -> 0°
      wait 1s
      SG90 -> 0°
    """
    try:
        # SG90 to 180
        smooth_move_servo(kit, CH_SG90, 180, steps=10, delay=0.03,
                          amin=SG90_MIN_ANGLE, amax=SG90_MAX_ANGLE)
        time.sleep(1.0)

        # MS24 to 15, hold, then back to 0
        smooth_move_servo(kit, CH_MS24, 15, steps=6, delay=0.03,
                          amin=MS24_MIN_ANGLE, amax=MS24_MAX_ANGLE)
        time.sleep(1.0)
        smooth_move_servo(kit, CH_MS24, 0, steps=6, delay=0.03,
                          amin=MS24_MIN_ANGLE, amax=MS24_MAX_ANGLE)

        # wait then SG90 back to 0
        time.sleep(1.0)
        smooth_move_servo(kit, CH_SG90, 0, steps=10, delay=0.03,
                          amin=SG90_MIN_ANGLE, amax=SG90_MAX_ANGLE)
    except Exception as e:
        print(f"[Servo] Sequence error: {e}", file=sys.stderr)

# -----------------------------
# Color matching helpers
# -----------------------------
def make_rgb_bounds_relative(target_rgb: Tuple[int, int, int], rel: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-channel ±(rel * channel_value) bounds (rounded to ints).
    Clipped to [0,255].
    """
    r, g, b = target_rgb
    rt = int(round(max(1, r * rel)))  # ensure at least ±1 where non-zero
    gt = int(round(max(1, g * rel))) if g > 0 else 1
    bt = int(round(max(1, b * rel))) if b > 0 else 1

    lower = np.array([max(r - rt, 0), max(g - gt, 0), max(b - bt, 0)], dtype=np.uint8)
    upper = np.array([min(r + rt, 255), min(g + gt, 255), min(b + bt, 255)], dtype=np.uint8)
    return lower, upper

def find_color_regions_rgb(frame_bgr: np.ndarray,
                           target_rgb: Tuple[int, int, int],
                           rel_tol: float,
                           area_min: int) -> Tuple[List[Tuple[int,int,int,int]], np.ndarray]:
    """
    Returns list of bounding boxes (x,y,w,h) for regions within the RGB window,
    and the 1-channel mask used.
    """
    # Convert BGR->RGB for intuitive matching
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    lower, upper = make_rgb_bounds_relative(target_rgb, rel_tol)

    # Mask and clean
    mask = cv2.inRange(frame_rgb, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, KERNEL, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Contours
    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_info) == 3:
        _, cnts, _ = contours_info
    else:
        cnts, _ = contours_info

    boxes = []
    for c in cnts:
        if cv2.contourArea(c) < area_min:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
    return boxes, mask

# -----------------------------
# Main loop
# -----------------------------
def annotate_frame(frame: np.ndarray, boxes: List[Tuple[int,int,int,int]], detected: bool):
    h, w = frame.shape[:2]
    # Draw boxes
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    # Status text
    label = "RUST DETECTED" if detected else "NO RUST DETECTED"
    color = (0, 0, 255) if detected else (255, 255, 255)
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Waterline bar to visualize detection strength (just count of boxes)
    cv2.rectangle(frame, (10, h - 20), (10 + min(len(boxes)*20, w - 20), h - 10), (0, 255, 0), -1)

def main():
    ap = argparse.ArgumentParser(description="Tight RGB rust detector + servo control")
    ap.add_argument("--rgb", type=int, nargs=3, default=list(TARGET_RGB_DEFAULT),
                    metavar=("R","G","B"),
                    help="Target RGB (default 213 59 119)")
    ap.add_argument("--tol", type=float, default=REL_TOL_DEFAULT,
                    help="Relative per-channel tolerance (e.g., 0.05 for ±5%% of each channel)")
    ap.add_argument("--area-min", type=int, default=AREA_MIN_DEFAULT,
                    help="Minimum contour area to accept as detection")
    ap.add_argument("--cam", type=int, default=CAM_INDEX_DEFAULT, help="Camera index (default 0)")
    ap.add_argument("--debounce-sec", type=float, default=DEBOUNCE_SEC_DEFAULT,
                    help="Minimum seconds between action sequences")
    ap.add_argument("--no-gui", action="store_true", help="Disable OpenCV window")
    args = ap.parse_args()

    target_rgb = tuple(int(clamp(v, 0, 255)) for v in args.rgb)
    lower, upper = make_rgb_bounds_relative(target_rgb, args.tol)

    print(f"[Config] Target RGB: {target_rgb}  tol={args.tol:.3f} "
          f"-> lower={lower.tolist()} upper={upper.tolist()}  area_min={args.area_min}")
    print(f"[Config] Debounce: {args.debounce_sec}s  Camera index: {args.cam}")

    # Initialize servos (required)
    kit = init_servos()

    # Camera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[Camera] Cannot open camera", file=sys.stderr)
        sys.exit(2)

    # Try reasonable defaults (tweak as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    last_fire = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[Camera] Empty frame, retrying...")
                time.sleep(0.05)
                continue

            boxes, mask = find_color_regions_rgb(frame, target_rgb, args.tol, args.area_min)
            detected = len(boxes) > 0

            # Annotate view
            annotate_frame(frame, boxes, detected)

            # Debounced trigger
            now = time.time()
            if detected and (now - last_fire) >= args.debounce_sec:
                print("[Detect] Rust match -> running sequence")
                run_sequence(kit)
                last_fire = time.time()

            if not args.no_gui:
                # Show side-by-side (frame + mask)
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                h1, w1 = frame.shape[:2]
                mask_bgr = cv2.resize(mask_bgr, (w1, h1))
                vis = np.hstack((frame, mask_bgr))
                cv2.imshow("Rust Detector (left=annotated, right=mask)", vis)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    print("[Exit] User requested quit.")
                    break
            else:
                # Headless: small sleep to keep CPU reasonable
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("[Exit] Ctrl+C received.")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        # Park servos
        try:
            smooth_move_servo(kit, CH_SG90, 0, steps=6, delay=0.03,
                              amin=SG90_MIN_ANGLE, amax=SG90_MAX_ANGLE)
            smooth_move_servo(kit, CH_MS24, 0, steps=6, delay=0.03,
                              amin=MS24_MIN_ANGLE, amax=MS24_MAX_ANGLE)
        except Exception:
            pass
        print("[Exit] Clean shutdown.")

if __name__ == "__main__":
    main()
