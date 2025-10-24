#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge Robot – Rust Detector (WEB PORTION UPDATED ONLY)

- Serves MJPEG at /stream (multipart/x-mixed-replace)
- Serves JSON at /status (for page stats)
- Uses a threaded HTTP server and ignores BrokenPipe on refreshes
- Binds to 127.0.0.1:8080 so nginx on :80 can reverse-proxy it

Everything else (camera, detection, color/tolerance, servos) remains as before.
"""

import argparse
import json
import sys
import time
import threading
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

# -----------------------------
# (UNCHANGED) Camera / stream
# -----------------------------
DEFAULT_HOST = "127.0.0.1"      # moved to localhost for nginx proxy
DEFAULT_PORT = 8080
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
USE_MJPG = True
JPEG_QUALITY = 40

# -----------------------------
# (UNCHANGED) Detection settings
# (Keep your current numbers; these are the same as your last run output)
# -----------------------------
TARGET_RGB = (195, 52, 110)
REL_TOL = 0.25
AREA_MIN = 600

# -----------------------------
# (UNCHANGED) Servos / PCA9685
# -----------------------------
PCA_CHANNELS = 16
PCA_ADDRESS = 0x40
PWM_FREQ = 50
CH_SG90 = 0
CH_MS24 = 1
SG90_RANGE = 180
SG90_PW = (500, 2500)
MS24_RANGE = 200
MS24_PW = (700, 2500)
SG90_IDLE_DEG = 180
SG90_FIRE_DEG = 0
MS24_FIRE_DEG = 20
LENS_DELAY = 1.0
SPRAY_TIME = 0.5
SMOOTH_STEPS = 12
SMOOTH_DELAY = 0.02
REARM_AFTER = 2.0

# -----------------------------
# (UNCHANGED) Helpers
# -----------------------------
def open_camera(idx, w, h, use_mjpg=True):
    backend = cv2.CAP_V4L2
    cap = cv2.VideoCapture(idx, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    for _ in range(5):
        cap.read()
    return cap

def make_rgb_bounds_relative(target_rgb, rel):
    r, g, b = target_rgb
    rt = int(round(max(1, r * rel))) if r > 0 else 1
    gt = int(round(max(1, g * rel))) if g > 0 else 1
    bt = int(round(max(1, b * rel))) if b > 0 else 1
    lower = np.array([max(r - rt, 0), max(g - gt, 0), max(b - bt, 0)], dtype=np.uint8)
    upper = np.array([min(r + rt, 255), min(g + gt, 255), min(b + bt, 255)], dtype=np.uint8)
    return lower, upper

def find_color_regions_rgb(frame_bgr, target_rgb, rel_tol, area_min=400):
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

# -----------------------------
# (UNCHANGED) Servos
# -----------------------------
class ServoController:
    def __init__(self):
        self.enabled = False
        try:
            from adafruit_servokit import ServoKit
            self.kit = ServoKit(channels=PCA_CHANNELS, address=PCA_ADDRESS)
            try:
                pca = getattr(self.kit, "_pca", None)
                if pca is not None:
                    pca.frequency = PWM_FREQ
            except Exception:
                pass
            self.kit.servo[CH_SG90].actuation_range = SG90_RANGE
            self.kit.servo[CH_SG90].set_pulse_width_range(*SG90_PW)
            self.kit.servo[CH_MS24].actuation_range = MS24_RANGE
            self.kit.servo[CH_MS24].set_pulse_width_range(*MS24_PW)
            self.kit.servo[CH_SG90].angle = SG90_IDLE_DEG
            self.kit.servo[CH_MS24].angle = 0
            self.enabled = True
        except Exception as e:
            print(f"[Servo] Disabled (init failed): {e}", file=sys.stderr)
            self.kit = None

    def _set(self, ch, deg):
        if not self.enabled: return
        try:
            self.kit.servo[ch].angle = float(deg)
        except Exception as e:
            print(f"[Servo] set ch={ch} failed: {e}", file=sys.stderr)

    def smooth_move(self, ch, target, steps=SMOOTH_STEPS, delay=SMOOTH_DELAY):
        if not self.enabled:
            return
        try:
            cur = self.kit.servo[ch].angle
        except Exception:
            cur = target
        if cur is None:
            cur = target
        cur = float(cur)
        target = float(target)
        if steps <= 1:
            self._set(ch, target); time.sleep(delay); return
        for i in range(1, steps + 1):
            a = cur + (target - cur) * (i / steps)
            self._set(ch, a); time.sleep(delay)

    def run_sequence(self):
        if not self.enabled:
            print("[Servo] Sequence skipped (servos disabled).")
            return
        self.smooth_move(CH_SG90, SG90_FIRE_DEG)
        time.sleep(LENS_DELAY)
        self.smooth_move(CH_MS24, MS24_FIRE_DEG)
        time.sleep(SPRAY_TIME)
        self.smooth_move(CH_MS24, 0)
        time.sleep(LENS_DELAY)
        self.smooth_move(CH_SG90, SG90_IDLE_DEG)

    def park(self):
        if not self.enabled: return
        try:
            self.smooth_move(CH_SG90, 0, steps=6, delay=0.03)
            self.smooth_move(CH_MS24, 0, steps=6, delay=0.03)
        except Exception:
            pass

# -----------------------------
# UPDATED WEB PORTION ONLY
# -----------------------------
class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class BackendHandler(BaseHTTPRequestHandler):
    frame_provider = None   # func -> latest JPEG bytes
    status_provider = None  # func -> dict

    def do_GET(self):
        path = self.path.split('?', 1)[0]

        if path == "/stream":
            try:
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                while True:
                    jpg = BackendHandler.frame_provider() if BackendHandler.frame_provider else b""
                    if not jpg:
                        time.sleep(0.02)
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                        self.wfile.write(jpg + b"\r\n")
                    except BrokenPipeError:
                        break
                return
            except BrokenPipeError:
                return
            except Exception:
                return

        if path == "/status":
            try:
                st = BackendHandler.status_provider() if BackendHandler.status_provider else {}
                payload = json.dumps(st, separators=(',', ':')).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except BrokenPipeError:
                    pass
                return
            except BrokenPipeError:
                return
            except Exception:
                try:
                    self.send_response(500); self.end_headers()
                except Exception:
                    pass
                return

        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        return

def start_backend(get_jpeg, get_status, host, port):
    BackendHandler.frame_provider = get_jpeg
    BackendHandler.status_provider = get_status
    srv = ThreadingHTTPServer((host, port), BackendHandler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv

# -----------------------------
# (UNCHANGED) Main loop
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Bridge Robot – Rust Detector (web updated)")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Bind host (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port (default 8080)")
    ap.add_argument("--cam", type=int, default=CAM_INDEX, help="Camera index")
    ap.add_argument("--area", type=int, default=AREA_MIN, help="Minimum blob area")
    ap.add_argument("--tol", type=float, default=REL_TOL, help="Per-channel relative tolerance")
    args = ap.parse_args()

    lower, upper = make_rgb_bounds_relative(TARGET_RGB, args.tol)
    print(f"Target RGB {TARGET_RGB} tol={args.tol:.3f} -> lower={lower.tolist()} upper={upper.tolist()}  area_min={args.area}")
    print(f"Backend at http://{args.host}:{args.port}  endpoints: /stream, /status")

    cap = open_camera(args.cam, FRAME_W, FRAME_H, USE_MJPG)
    servos = ServoController()

    jpeg_lock = threading.Lock()
    latest_jpeg = bytearray()
    status = {
        "label": "INIT",
        "fps": 0.0,
        "last_boxes": 0,
        "target_rgb": list(TARGET_RGB),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "area_min": args.area,
    }

    def get_jpeg():
        with jpeg_lock:
            return bytes(latest_jpeg)

    def get_status():
        return dict(status)

    srv = start_backend(get_jpeg, get_status, host=args.host, port=args.port)

    last_trigger_ok = 0.0
    fps_prev = time.monotonic()
    fps_frames = 0
    fps_val = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01); continue

            boxes, mask = find_color_regions_rgb(frame, TARGET_RGB, args.tol, area_min=args.area)
            detected = len(boxes) > 0

            label = "RUST DETECTED" if detected else "NO RUST DETECTED"
            if detected:
                for (x, y, w, h) in boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)

            fps_frames += 1
            now = time.monotonic()
            if now - fps_prev >= 1.0:
                fps_val = fps_frames / (now - fps_prev)
                fps_prev, fps_frames = now, 0
            cv2.putText(frame, f"FPS: {fps_val:.1f}", (16, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok_jpg:
                with jpeg_lock:
                    latest_jpeg[:] = buf.tobytes()

            status["last_boxes"] = len(boxes)
            status["fps"] = fps_val
            status["label"] = label

            if detected and (now - last_trigger_ok) >= REARM_AFTER:
                try:
                    servos.run_sequence()
                except Exception as e:
                    print(f"[Servo] sequence error: {e}", file=sys.stderr)
                last_trigger_ok = time.monotonic()

    except KeyboardInterrupt:
        print("\n[Exit] Ctrl+C")
    finally:
        try: cap.release()
        except Exception: pass
        servos.park()
        print("[Exit] Clean shutdown.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
