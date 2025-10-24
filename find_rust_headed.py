#!/usr/bin/env python3
"""
Headless (web) rust detection + servo control for Raspberry Pi OS Lite.
- Live MJPEG preview at http://0.0.0.0:8080/ (and /stream)
- Servos REQUIRED (no --no-servos mode). If servo init fails, program exits.
- Handles HSV wrap-around. Smooth servo moves.

Install:
  # in your venv
  pip install --no-cache-dir opencv-python adafruit-blinka adafruit-circuitpython-servokit
  sudo apt update && sudo apt install -y python3-smbus i2c-tools
  sudo raspi-config  # Interface Options -> I2C -> Enable

Run:
  python3 find_rust_headed.py
  # open http://129.21.151.183:8080/ in a browser
"""
import argparse
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

try:
    from adafruit_servokit import ServoKit
except Exception as e:
    print("ERROR: adafruit-circuitpython-servokit not available. Install it and enable I2C.", file=sys.stderr)
    raise

# ================== CONFIG ==================
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

MS24_RANGE = 200
MS24_PW = (700, 2500)

SG90_DEG = 180
MS24_DEG = 20

REARM_AFTER = 2.0
SMOOTH_STEPS = 12
SMOOTH_DELAY = 0.02

LENS_DELAY = 1.0
SPRAY_TIME = 0.5

WEB_PREVIEW_PORT = 8080
# ============================================


def rgb_to_hsv_center(rgb):
    r, g, b = rgb
    swatch = np.uint8([[[b, g, r]]])  # BGR for OpenCV
    hsv = cv2.cvtColor(swatch, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def hsv_bounds_from_center(h_center, h_tol, s_min, v_min):
    low_h = h_center - h_tol
    high_h = h_center + h_tol
    if low_h < 0 or high_h > 179:
        return None, (np.array([0, s_min, v_min], dtype=np.uint8),
                      np.array([179, 255, 255], dtype=np.uint8)), (low_h, high_h)
    lower = np.array([max(low_h, 0), s_min, v_min], dtype=np.uint8)
    upper = np.array([min(high_h, 179), 255, 255], dtype=np.uint8)
    return (lower, upper), None, None


def open_camera(idx, w, h, use_mjpg=True):
    backend = cv2.CAP_V4L2
    cap = cv2.VideoCapture(idx, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try a different index or check v4l2-ctl.")
    for _ in range(5):
        cap.read()
    return cap


def find_color_regions(frame_bgr, bounds_pair, wrap_params, area_min=400):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    plain_bounds, _ = bounds_pair

    if plain_bounds is not None:
        lower, upper = plain_bounds
        mask = cv2.inRange(hsv, lower, upper)
    else:
        if wrap_params is None:
            return [], None
        low_h, high_h = wrap_params
        upper1 = np.array([min(high_h, 179), 255, 255], dtype=np.uint8)
        lower1 = np.array([0, S_MIN, V_MIN], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        lower2 = np.array([max(low_h, 0), S_MIN, V_MIN], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

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


def init_servos():
    kit = ServoKit(channels=PCA_CHANNELS, address=PCA_ADDRESS)
    try:
        pca = getattr(kit, "_pca", None)
        if pca is not None:
            pca.frequency = PWM_FREQ
    except Exception as e:
        print(f"Warning: failed to set PWM frequency: {e}")
    try:
        kit.servo[CH_SG90].actuation_range = SG90_RANGE
        kit.servo[CH_SG90].set_pulse_width_range(*SG90_PW)
        kit.servo[CH_MS24].actuation_range = MS24_RANGE
        kit.servo[CH_MS24].set_pulse_width_range(*MS24_PW)
        kit.servo[CH_SG90].angle = SG90_DEG
        kit.servo[CH_MS24].angle = 0
    except Exception as e:
        print(f"Warning configuring servos: {e}")
    time.sleep(0.3)
    return kit


def smooth_move_servo(kit, channel, target_deg, steps=SMOOTH_STEPS, delay=SMOOTH_DELAY):
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
    smooth_move_servo(kit, CH_SG90, 0)
    time.sleep(LENS_DELAY)
    smooth_move_servo(kit, CH_MS24, MS24_DEG)
    time.sleep(SPRAY_TIME)
    smooth_move_servo(kit, CH_MS24, 0)
    time.sleep(LENS_DELAY)
    smooth_move_servo(kit, CH_SG90, SG90_DEG)


# ---------------- MJPEG Web Server ----------------
class _MJPEGHandler(BaseHTTPRequestHandler):
    frame_provider = None  # func -> latest JPEG bytes
    page_provider = None   # func -> HTML bytes

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = _MJPEGHandler.frame_provider() if _MJPEGHandler.frame_provider else b""
                    if not jpg:
                        time.sleep(0.02); continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                    self.wfile.write(jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            html = _MJPEGHandler.page_provider().decode("utf-8") if _MJPEGHandler.page_provider else "<h3>OK</h3>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

    def log_message(self, fmt, *args):
        return


def start_mjpeg_server(get_jpeg_func, get_page_func, port=WEB_PREVIEW_PORT):
    _MJPEGHandler.frame_provider = get_jpeg_func
    _MJPEGHandler.page_provider = get_page_func
    srv = HTTPServer(("129.21.151.183", port), _MJPEGHandler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv


def main(argv=None):
    parser = argparse.ArgumentParser(description="Web-headed rust detection + servo control (Pi Lite)")
    parser.add_argument("--cam", type=int, default=CAM_INDEX, help="Camera index (default 0)")
    parser.add_argument("--w", type=int, default=FRAME_W, help="Frame width")
    parser.add_argument("--h", type=int, default=FRAME_H, help="Frame height")
    parser.add_argument("--area", type=int, default=AREA_MIN, help="Min contour area")
    parser.add_argument("--rgb", nargs=3, type=int, metavar=("R", "G", "B"),
                        default=list(TARGET_RGB), help="Target RGB (3 values)")
    parser.add_argument("--port", type=int, default=WEB_PREVIEW_PORT, help="Web preview port")
    args = parser.parse_args(argv)

    # Color band
    target_rgb = tuple(args.rgb)
    h_center, s_center, v_center = rgb_to_hsv_center(target_rgb)
    lower_upper, wrap_upper, wrap_params = hsv_bounds_from_center(h_center, H_TOL, S_MIN, V_MIN)
    print(f"Target RGB {target_rgb} -> HSV center (OpenCV) = {(h_center, s_center, v_center)}")
    if lower_upper is not None:
        lo, up = lower_upper
        print(f"Using HSV lower={lo.tolist()} upper={up.tolist()}")
    else:
        print(f"Hue range wraps {wrap_params}; using two-range mask.")

    # Camera
    cap = open_camera(args.cam, args.w, args.h, USE_MJPG)

    # Servos (REQUIRED)
    try:
        kit = init_servos()
    except Exception as e:
        print(f"ERROR: Servo init failed: {e}", file=sys.stderr)
        sys.exit(2)

    # Web preview
    jpeg_lock = threading.Lock()
    latest_jpeg = bytearray()
    status = {"label": "INIT", "fps": 0.0, "last_boxes": 0}

    def get_jpeg():
        with jpeg_lock:
            return bytes(latest_jpeg)

    def get_page():
        # simple status page with embedded MJPEG
        html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Rust Detector</title>
  <style>
    body {{ font-family: sans-serif; background:#111; color:#eee; margin:0; }}
    header {{ padding:12px 16px; background:#222; position:sticky; top:0; }}
    .row {{ display:flex; gap:12px; padding:12px; align-items:flex-start; }}
    .card {{ background:#1d1d1d; border-radius:10px; padding:12px; }}
    .stat {{ font-size:14px; margin:4px 0; }}
    img {{ max-width:100%; height:auto; border-radius:10px; }}
    code {{ background:#333; padding:2px 6px; border-radius:4px; }}
  </style>
</head>
<body>
<header>
  <strong>Rust Detector</strong>
  <span style="margin-left:16px;">Status: <code>{status["label"]}</code></span>
  <span style="margin-left:16px;">FPS: <code>{status["fps"]:.1f}</code></span>
  <span style="margin-left:16px;">Detections: <code>{status["last_boxes"]}</code></span>
</header>
<div class="row">
  <div class="card">
    <img src="/stream" />
  </div>
  <div class="card">
    <div class="stat">HSV center: <code>({h_center}, {s_center}, {v_center})</code></div>
    <div class="stat">H_TOL: <code>{H_TOL}</code> S_MIN: <code>{S_MIN}</code> V_MIN: <code>{V_MIN}</code></div>
    <div class="stat">Area min: <code>{args.area}</code></div>
    <div class="stat">Servos: <code>ENABLED</code> (SG90 ch {CH_SG90}, MS24 ch {CH_MS24})</div>
    <div class="stat">Sequence: SG90→{SG90_DEG}°, MS24→{MS24_DEG}°, return</div>
  </div>
</div>
</body>
</html>
""".strip().encode("utf-8")
        return html

    srv = start_mjpeg_server(get_jpeg, get_page, port=args.port)
    print(f"Web preview at http://129.21.151.183:{args.port}/ (stream at /stream)")

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

            boxes, mask = find_color_regions(frame, (lower_upper, wrap_upper), wrap_params, area_min=args.area)
            detected = len(boxes) > 0

            # Overlay
            if detected:
                for (x, y, w, h) in boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                label = "RUST DETECTED"
            else:
                label = "NO RUST DETECTED"

            cv2.putText(frame, label, (16, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            # FPS overlay
            fps_frames += 1
            now = time.monotonic()
            if now - fps_prev >= 1.0:
                fps_val = fps_frames / (now - fps_prev)
                fps_prev, fps_frames = now, 0
            cv2.putText(frame, f"FPS: {fps_val:.1f}", (16, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Push JPEG
            ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok_jpg:
                with jpeg_lock:
                    latest_jpeg[:] = buf.tobytes()

            # Trigger logic
            status["last_boxes"] = len(boxes)
            status["fps"] = fps_val
            status["label"] = label

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
        try:
            smooth_move_servo(kit, CH_SG90, 0, steps=6, delay=0.03)
            smooth_move_servo(kit, CH_MS24, 0, steps=6, delay=0.03)
        except Exception:
            pass
        cap.release()
        print("Exited cleanly.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
