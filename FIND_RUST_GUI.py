#!/usr/bin/env python3
"""
Improved rust-find + servo control script.

Main changes from original:
- Fixed fullscreen toggle syntax error.
- Added --no-servos mode for development without I2C hardware.
- Handles HSV hue wrap-around (near 0/179).
- Uses time.monotonic() for timing.
- Adds smoother servo motion to reduce jerk.
- Adds CLI for common runtime tweaks and optional mask display for calibration.
- Adds a simple click-to-sample color calibration (press 'c' in preview).
"""
import argparse
import sys
import time
import math

import cv2
import numpy as np

# Optional Adafruit servo lib; lazily imported when needed
try:
    from adafruit_servokit import ServoKit
except Exception:
    ServoKit = None

# ================== USER CONFIG (defaults, can be overridden via CLI) ==================
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
USE_MJPG = True

TARGET_RGB = (213, 59, 119)

H_TOL = 12
S_MIN = 90
V_MIN = 80
AREA_MIN = 600

PCA_CHANNELS = 16
PCA_ADDRESS = 0x40
PWM_FREQ = 50

CH_SG90 = 0
CH_MS24 = 1

SG90_RANGE = 180
SG90_PW = (500, 2500)

MS24_RANGE = 180
MS24_PW = (500, 2500)

SG90_DEG = 180
MS24_DEG = 15

REARM_AFTER = 2.0  # seconds after sequence end
SMOOTH_STEPS = 12
SMOOTH_DELAY = 0.02  # seconds between small servo steps
# ================= END CONFIG ==================


def rgb_to_hsv_center(rgb):
    # Convert BGR triplet for OpenCV and return center HSV (h,s,v)
    r, g, b = rgb
    swatch = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(swatch, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def hsv_bounds_from_center(h_center, h_tol, s_min, v_min):
    # Return lower/upper arrays, but indicate whether wrap-around occurs
    low_h = h_center - h_tol
    high_h = h_center + h_tol
    if low_h < 0 or high_h > 179:
        # wrap-around: we will produce two ranges
        return None, (np.array([0, s_min, v_min], dtype=np.uint8),
                      np.array([179, 255, 255], dtype=np.uint8)), (low_h, high_h)
    else:
        lower = np.array([max(low_h, 0), s_min, v_min], dtype=np.uint8)
        upper = np.array([min(high_h, 179), 255, 255], dtype=np.uint8)
        return (lower, upper), None, None


def open_camera(idx, w, h, use_mjpg=True):
    # Use V4L2 backend on Linux for better control if available
    backend = cv2.CAP_V4L2
    cap = cv2.VideoCapture(idx, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try a different index or check v4l2-ctl.")
    # Warm up a few frames
    for _ in range(5):
        cap.read()
    return cap


def find_color_regions(frame_bgr, lower_upper, wrap_params, area_min=400):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if lower_upper[0] is not None:
        lower, upper = lower_upper[0]
        mask = cv2.inRange(hsv, lower, upper)
    else:
        # wrap-around: combine two ranges
        low_wrap, high_wrap = lower_upper[1]
        # low_wrap and high_wrap here are the extrema; but we need to build two masks
        # compute the two masks using wrap_params (low_h, high_h)
        low_h, high_h = wrap_params
        # mask1: 0..high_h
        upper1 = np.array([min(high_h, 179), 255, 255], dtype=np.uint8)
        lower1 = np.array([0, S_MIN, V_MIN], dtype=np.uint8)
        # mask2: low_h..179
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        lower2 = np.array([max(low_h, 0), S_MIN, V_MIN], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    # Optional small blur to reduce speckle
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # findContours compatibility across OpenCV versions
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


def init_servos(no_servos=False):
    if no_servos:
        print("Running in no-servos mode (simulation).")
        return None
    if ServoKit is None:
        raise RuntimeError("adafruit_servokit not available; install adafruit-circuitpython-servokit or run --no-servos.")
    kit = ServoKit(channels=PCA_CHANNELS, address=PCA_ADDRESS, frequency=PWM_FREQ)
    # Configure pulse ranges and actuation spans
    try:
        kit.servo[CH_SG90].actuation_range = SG90_RANGE
        kit.servo[CH_SG90].set_pulse_width_range(*SG90_PW)
        kit.servo[CH_MS24].actuation_range = MS24_RANGE
        kit.servo[CH_MS24].set_pulse_width_range(*MS24_PW)
    except Exception as e:
        print(f"Warning: failed to set actuation/pulse ranges: {e}")
    # Initialize to 0 degrees if possible
    try:
        kit.servo[CH_SG90].angle = 0
        kit.servo[CH_MS24].angle = 0
    except Exception:
        pass
    time.sleep(0.3)
    return kit


def smooth_move_servo(kit, channel, target_deg, steps=SMOOTH_STEPS, delay=SMOOTH_DELAY):
    if kit is None:
        print(f"[sim] servo {channel} -> {target_deg}°")
        return
    try:
        cur = kit.servo[channel].angle
    except Exception:
        cur = target_deg
    if cur is None:
        cur = target_deg
    cur = float(cur)
    target = float(target_deg)
    if steps <= 1:
        try:
            kit.servo[channel].angle = target
        except Exception:
            pass
        return
    for i in range(1, steps + 1):
        a = cur + (target - cur) * (i / steps)
        try:
            kit.servo[channel].angle = float(a)
        except Exception:
            pass
        time.sleep(delay)


def rust_sequence(kit):
    # SG90: 0 -> SG90_DEG
    smooth_move_servo(kit, CH_SG90, SG90_DEG)
    time.sleep(1.0)
    # MS24: 0 -> MS24_DEG
    smooth_move_servo(kit, CH_MS24, MS24_DEG)
    time.sleep(1.0)
    # MS24: back to 0
    smooth_move_servo(kit, CH_MS24, 0)
    time.sleep(1.0)
    # SG90: back to 0
    smooth_move_servo(kit, CH_SG90, 0)


def sample_color_at_click(event, x, y, flags, param):
    # callback to pick the color under mouse (BGR)
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    frame = param.get("frame")
    if frame is None:
        return
    bgr = frame[y, x].tolist()
    print(f"Sampled BGR at ({x},{y}) = {bgr}")
    param["sampled_bgr"] = bgr


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rust detection + servo demo")
    parser.add_argument("--cam", type=int, default=CAM_INDEX, help="Camera index (default 0)")
    parser.add_argument("--w", type=int, default=FRAME_W, help="Frame width")
    parser.add_argument("--h", type=int, default=FRAME_H, help="Frame height")
    parser.add_argument("--no-servos", action="store_true", help="Run without servos (simulation)")
    parser.add_argument("--area", type=int, default=AREA_MIN, help="Min contour area")
    parser.add_argument("--show-mask", action="store_true", help="Also show mask window for calibration")
    parser.add_argument("--rgb", nargs=3, type=int, metavar=("R", "G", "B"),
                        default=list(TARGET_RGB), help="Target RGB (3 values)")
    args = parser.parse_args(argv)

    target_rgb = tuple(args.rgb)
    h_center, s_center, v_center = rgb_to_hsv_center(target_rgb)
    lower_upper, wrap_upper, wrap_params = hsv_bounds_from_center(h_center, H_TOL, S_MIN, V_MIN)
    print(f"Target RGB {target_rgb} -> HSV center (OpenCV) = {(h_center, s_center, v_center)}")
    if lower_upper[0] is not None:
        print(f"Using HSV lower={lower_upper[0].tolist()} upper={lower_upper[1].tolist()}")
    else:
        print("Hue range wraps. Using two-range mask.")

    cap = open_camera(args.cam, args.w, args.h, USE_MJPG)
    kit = None
    try:
        kit = init_servos(no_servos=args.no_servos)
    except Exception as e:
        print(f"Servo init failed: {e}\nContinuing in no-servos mode.")
        kit = None

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)

    show_mask = args.show_mask
    if show_mask:
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

    clicked = {"frame": None, "sampled_bgr": None}
    cv2.setMouseCallback("Camera", sample_color_at_click, clicked)

    triggered = False
    next_arm_time = 0.0

    fps_prev = time.monotonic()
    fps_frames = 0
    fps_val = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame grab failed")
                break

            clicked["frame"] = frame
            boxes, mask = find_color_regions(frame, (lower_upper, wrap_upper), wrap_params, area_min=args.area)
            detected = len(boxes) > 0

            # Draw detections
            if detected:
                for (x, y, w, h) in boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                label = "RUST DETECTED"
            else:
                label = "NO RUST DETECTED"

            cv2.putText(frame, label, (16, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            # FPS
            fps_frames += 1
            now = time.monotonic()
            if now - fps_prev >= 1.0:
                fps_val = fps_frames / (now - fps_prev)
                fps_prev, fps_frames = now, 0
            cv2.putText(frame, f"FPS: {fps_val:.1f}", (16, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Camera", frame)
            if show_mask and mask is not None:
                cv2.imshow("Mask", mask)

            k = cv2.waitKey(10) & 0xFF
            if k == ord('q'):
                break
            if k == ord('f'):
                # Toggle fullscreen
                fs = int(cv2.getWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN))
                cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL if fs == 1 else cv2.WINDOW_FULLSCREEN)
            if k == ord('c'):
                # If user clicked recently to sample color, print hsv and suggestion
                if clicked.get("sampled_bgr") is not None:
                    b, g, r = clicked["sampled_bgr"]
                    print(f"Sampled BGR: {(b, g, r)} -> set --rgb {r} {g} {b} to use")
                    clicked["sampled_bgr"] = None

            # Trigger logic
            if detected and (not triggered) and (time.monotonic() >= next_arm_time):
                triggered = True
                try:
                    rust_sequence(kit)
                except Exception as e:
                    print(f"Error during rust sequence: {e}")
                triggered = False
                next_arm_time = time.monotonic() + REARM_AFTER

    except KeyboardInterrupt:
        pass
    finally:
        # Safe shutdown
        if kit is not None:
            try:
                smooth_move_servo(kit, CH_SG90, 0, steps=6, delay=0.03)
                smooth_move_servo(kit, CH_MS24, 0, steps=6, delay=0.03)
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        print("Exited cleanly.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
