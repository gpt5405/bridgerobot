#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge Robot – Rust Detector (TurboJPEG Streaming)

WHAT'S CHANGED (performance only):
- Streaming encoder now uses TurboJPEG for /stream (much faster).
- Falls back to cv2.imencode if PyTurboJPEG or libturbojpeg are unavailable.
- Threaded HTTP server remains; endpoints: /stream (MJPEG), /status (JSON).

Everything else (color/detection/servos) is unchanged in behavior.

Run:
  python3 rustFindv4.py --host 127.0.0.1 --port 8080 --cam 0
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
# Camera / stream settings
# -----------------------------
DEFAULT_HOST = "127.0.0.1"      # bind backend to localhost; nginx proxies from :80
DEFAULT_PORT = 8080
CAM_INDEX    = 0
FRAME_W, FRAME_H = 1280, 720
USE_MJPG     = True             # request MJPG from the camera for capture
REQ_FPS      = 60               # ask the camera for 60 FPS

# Streaming encode quality
TURBOJPEG_QUALITY = 60          # great quality/speed tradeoff
OPENCV_QUALITY    = 60          # fallback if TurboJPEG is unavailable

# Optional: stream a smaller preview to cut encode time (keeps detection at full res)
STREAM_DOWNSCALE_WIDTH = 0      # set e.g. 854 for 480p preview; 0 = disabled

# -----------------------------
# Detection settings (UNCHANGED)
# -----------------------------
TARGET_RGB = (195, 52, 110)     # leave as-is (your chosen target)
REL_TOL    = 0.25               # leave as-is
AREA_MIN   = 600

# -----------------------------
# Servo / PCA9685 (UNCHANGED)
# -----------------------------
PCA_CHANNELS = 16
PCA_ADDRESS  = 0x40
PWM_FREQ     = 50
CH_SG90, CH_MS24 = 0, 1
SG90_RANGE, SG90_PW = 180, (500, 2500)
MS24_RANGE, MS24_PW = 200, (700, 2500)
SG90_IDLE_DEG, SG90_FIRE_DEG = 180, 0
MS24_FIRE_DEG = 20
LENS_DELAY, SPRAY_TIME = 1.0, 0.5
SMOOTH_STEPS, SMOOTH_DELAY = 12, 0.02
REARM_AFTER = 2.0

# -----------------------------
# TurboJPEG setup (encoder)
# -----------------------------
class JpegEncoder:
    def __init__(self):
        self.backend = "opencv"
        self.jq = OPENCV_QUALITY
        self.jpeg = None
        try:
            from turbojpeg import TurboJPEG, TJPF_BGR
            self.jpeg = TurboJPEG()          # uses system libturbojpeg
            self.TJPF_BGR = TJPF_BGR
            self.backend = "turbojpeg"
            self.jq = TURBOJPEG_QUALITY
            print("[Encoder] Using TurboJPEG")
        except Exception as e:
            print(f"[Encoder] TurboJPEG unavailable, falling back to OpenCV: {e}", file=sys.stderr)

    def encode(self, frame_bgr) -> bytes:
        """
        Return JPEG bytes for the given BGR frame.
        If STREAM_DOWNSCALE_WIDTH is set, downscale before encoding.
        """
        img = frame_bgr
        if STREAM_DOWNSCALE_WIDTH and STREAM_DOWNSCALE_WIDTH > 0:
            h, w = frame_bgr.shape[:2]
            if w > STREAM_DOWNSCALE_WIDTH:
                new_w = STREAM_DOWNSCALE_WIDTH
                new_h = int(h * (new_w / float(w)))
                img = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if self.backend == "turbojpeg" and self.jpeg is not None:
            return self.jpeg.encode(img, quality=self.jq, pixel_format=self.TJPF_BGR)
        else:
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jq)])
            if not ok:
                return b""
            return buf.tobytes()

# -----------------------------
# Helpers (UNCHANGED logic)
# -----------------------------
def open_camera(idx, w, h, use_mjpg=True):
    backend = cv2.CAP_V4L2
    cap = cv2.VideoCapture(idx, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # Ask for FPS (camera may clamp to supported modes)
    cap.set(cv2.CAP_PROP_FPS, REQ_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    # Warm up
    for _ in range(5):
        cap.read()
    # Print negotiated FPS for visibility
    try:
        got_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[Camera] Negotiated FPS: {got_fps:.1f}")
    except Exception:
        pass
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
        cur = float(cur); target = float(target)
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
# Threaded backend HTTP (stream+status)
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
                self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                while True:
                    jpg = BackendHandler.frame_provider() if BackendHandler.frame_provider else b""
                    if not jpg:
                        time.sleep(0.01)
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
# Main loop
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Bridge Robot – Rust Detector (TurboJPEG)")
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
    encoder = JpegEncoder()

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
        "encoder": encoder.backend,
    }

    def get_jpeg():
        with jpeg_lock:
            return bytes(latest_jpeg)

    def get_status():
        return dict(status)

    _ = start_backend(get_jpeg, get_status, host=args.host, port=args.port)

    last_trigger_ok = 0.0
    fps_prev = time.monotonic()
    fps_frames = 0
    fps_val = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            boxes, mask = find_color_regions_rgb(frame, TARGET_RGB, args.tol, area_min=args.area)
            detected = len(boxes) > 0

            # Annotate preview
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

            # Encode for stream (TurboJPEG or OpenCV)
            jpeg_bytes = encoder.encode(frame)
            if jpeg_bytes:
                with jpeg_lock:
                    latest_jpeg[:] = jpeg_bytes

            # Update status
            status["last_boxes"] = len(boxes)
            status["fps"] = fps_val
            status["label"] = label

            # Debounced action
            if detected and (now - last_trigger_ok) >= REARM_AFTER:
                try:
                    servos.run_sequence()
                except Exception as e:
                    print(f"[Servo] sequence error: {e}", file=sys.stderr)
                last_trigger_ok = time.monotonic()

    except KeyboardInterrupt:
        print("\n[Exit] Ctrl+C")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        servos.park()
        print("[Exit] Clean shutdown.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
