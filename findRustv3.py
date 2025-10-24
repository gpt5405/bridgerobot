#!/usr/bin/env python3
"""
Web-headed rust detection + servo control (tight RGB matching).

- Live MJPEG preview at http://129.21.151.183:8080/ (and /stream)
- Tight RGB match around TARGET_RGB with per-channel ±relative tolerance.
- Keeps your existing servo sequence and web UI flow.

Install (venv recommended):
  pip install --no-cache-dir opencv-python adafruit-blinka adafruit-circuitpython-servokit
  sudo apt update && sudo apt install -y python3-smbus i2c-tools
  sudo raspi-config  # Interface Options -> I2C -> Enable
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

# Target and tolerance (per-channel relative, e.g., 0.25 = ±25% of channel value)
TARGET_RGB = (195, 52, 110)
REL_TOL = 0.25

AREA_MIN = 600  # contour area threshold

# PCA9685 / servos (kept consistent with your setup)
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

WEB_HOST = "129.21.151.183"  # matches your existing binding
WEB_PREVIEW_PORT = 8080
# ============================================


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
    # warm up a few frames
    for _ in range(5):
        cap.read()
    return cap


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
    # SG90 lens -> 0, delay, MS24 spray on/off, delay, SG90 lens -> SG90_DEG
    smooth_move_servo(kit, CH_SG90, 0)
    time.sleep(LENS_DELAY)
    smooth_move_servo(kit, CH_MS24, MS24_DEG)
    time.sleep(SPRAY_TIME)
    smooth_move_servo(kit, CH_MS24, 0)
    time.sleep(LENS_DELAY)
    smooth_move_servo(kit, CH_SG90, SG90_DEG)


# ---------------- RGB tight matching ----------------
def make_rgb_bounds_relative(target_rgb, rel):
    r, g, b = target_rgb
    rt = int(round(max(1, r * rel))) if r > 0 else 1
    gt = int(round(max(1, g * rel))) if g > 0 else 1
    bt = int(round(max(1, b * rel))) if b > 0 else 1
    lower = np.array([max(r - rt, 0), max(g - gt, 0), max(b - bt, 0)], dtype=np.uint8)
    upper = np.array([min(r + rt, 255), min(g + gt, 255), min(b + bt, 255)], dtype=np.uint8)
    return lower, upper


def find_color_regions_rgb(frame_bgr, target_rgb, rel_tol, area_min=400):
    """Return (boxes, mask) for pixels inside the tight RGB window."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    lower, upper = make_rgb_bounds_relative(target_rgb, rel_tol)
    mask = cv2.inRange(frame_rgb, lower, upper)

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


def start_mjpeg_server(get_jpeg_func, get_page_func, host, port):
    _MJPEGHandler.frame_provider = get_jpeg_func
    _MJPEGHandler.page_provider = get_page_func
    srv = HTTPServer((host, port), _MJPEGHandler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv


def main(argv=None):
    p = argparse.ArgumentParser(description="Web-headed rust detection + servo control (tight RGB)")
    p.add_argument("--cam", type=int, default=CAM_INDEX, help="Camera index (default 0)")
    p.add_argument("--w", type=int, default=FRAME_W, help="Frame width")
    p.add_argument("--h", type=int, default=FRAME_H, help="Frame height")
    p.add_argument("--area", type=int, default=AREA_MIN, help="Min contour area")
    p.add_argument("--rgb", nargs=3, type=int, metavar=("R", "G", "B"),
                   default=list(TARGET_RGB), help="Target RGB (3 values)")
    p.add_argument("--tol", type=float, default=REL_TOL,
                   help="Per-channel relative tolerance (e.g., 0.05 means ±5% of each channel)")
    p.add_argument("--port", type=int, default=WEB_PREVIEW_PORT, help="Web preview port")
    p.add_argument("--host", type=str, default=WEB_HOST, help="Bind host for web server")
    args = p.parse_args(argv)

    target_rgb = tuple(int(max(0, min(255, v))) for v in args.rgb)
    lower, upper = make_rgb_bounds_relative(target_rgb, args.tol)
    print(f"Target RGB {target_rgb} with tol={args.tol:.3f} -> lower={lower.tolist()} upper={upper.tolist()}  area_min={args.area}")

    # Camera
    cap = open_camera(args.cam, args.w, args.h, USE_MJPG)

    # Servos (required)
    try:
        kit = init_servos()
    except Exception as e:
        print(f"ERROR: Servo init failed: {e}", file=sys.stderr)
        sys.exit(2)

    # Web preview state
    jpeg_lock = threading.Lock()
    latest_jpeg = bytearray()
    status = {"label": "INIT", "fps": 0.0, "last_boxes": 0}

    def get_jpeg():
        with jpeg_lock:
            return bytes(latest_jpeg)

    def get_page():
        html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Rust Detector (RGB)</title>
  <style>
    body {{ font-family: sans-serif; background:#111; color:#eee; margin:0; }}
    header {{ padding:12px 16px; background:#222; position:sticky; top:0; }}
    .row {{ display:flex; gap:12px; padding:12px; align-items:flex-start; flex-wrap:wrap; }}
    .card {{ background:#1d1d1d; border-radius:10px; padding:12px; }}
    .stat {{ font-size:14px; margin:4px 0; }}
    img {{ max-width:100%; height:auto; border-radius:10px; }}
    code {{ background:#333; padding:2px 6px; border-radius:4px; }}
  </style>
</head>
<body>
<header>
  <strong>Rust Detector (RGB)</strong>
  <span style="margin-left:16px;">Status: <code>{status["label"]}</code></span>
  <span style="margin-left:16px;">FPS: <code>{status["fps"]:.1f}</code></span>
  <span style="margin-left:16px;">Detections: <code>{status["last_boxes"]}</code></span>
</header>
<div class="row">
  <div class="card"><img src="/stream" /></div>
  <div class="card">
    <div class="stat">Target RGB: <code>{target_rgb}</code></div>
    <div class="stat">Window lower: <code>{tuple(lower.tolist())}</code></div>
    <div class="stat">Window upper: <code>{tuple(upper.tolist())}</code></div>
    <div class="stat">Area min: <code>{args.area}</code></div>
    <div class="stat">Servos: <code>ENABLED</code> (SG90 ch {CH_SG90}, MS24 ch {CH_MS24})</div>
    <div class="stat">Sequence: SG90→{SG90_DEG}°, MS24→{MS24_DEG}°, return</div>
  </div>
</div>
</body>
</html>
""".strip().encode("utf-8")
        return html

    srv = start_mjpeg_server(get_jpeg, get_page, host=args.host, port=args.port)
    print(f"Web preview at http://{args.host}:{args.port}/ (stream at /stream)")

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

            boxes, mask = find_color_regions_rgb(frame, target_rgb, args.tol, area_min=args.area)
            detected = len(boxes) > 0

            # Draw
            label = "RUST DETECTED" if detected else "NO RUST DETECTED"
            if detected:
                for (x, y, w, h) in boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # FPS calc
            fps_frames += 1
            now = time.monotonic()
            if now - fps_prev >= 1.0:
                fps_val = fps_frames / (now - fps_prev)
                fps_prev, fps_frames = now, 0
            cv2.putText(frame, f"FPS: {fps_val:.1f}", (16, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # JPEG push
            ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok_jpg:
                with jpeg_lock:
                    latest_jpeg[:] = buf.tobytes()

            # Update status + trigger
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
