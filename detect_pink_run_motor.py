import cv2
import numpy as np
from time import sleep
import board
import busio
from adafruit_pca9685 import PCA9685

# Setup I2C and PCA9685

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50 # Standard Servo Frequency

# Servo Channels

SG90_CH = 0
SERVO20KG_CH = 1

# Helper: set pulse width (0.5 ms to 2.5 ms mapped to 0-4095)

def set_servo_angle(channel, angle):
	pulse_min = 1000  # 0.5 ms

	pulse_max = 5000 # 2.5 ms

	angle = max(0, min(180, angle))
	pulse = int(pulse_min + (pulse_max - pulse_min) * angle / 180)
	pca.channels[channel].duty_cycle = pulse

# Capture Frame

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
	print("Failed to capture image")
	exit()

# Convert to HSV

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
target_rgb = np.uint8([[[213, 59, 119]]])
target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV)[0][0]
hue, sat, val = map(int, target_hsv)

lower = np.array([max(hue - 10, 0), max(sat - 50, 0), max(val - 50, 0)])
upper = np.array([min(hue + 10, 179), min(sat + 50, 255), min(val + 50, 255)])

mask = cv2.inRange(hsv, lower, upper)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rust_found = False
for cnt in contours:
	area = cv2.contourArea(cnt)
	if area > 100:
		x, y, w, h = cv2.boundingRect(cnt)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		rust_found = True
if rust_found:
	print("Rust Detected!")

# Actuate SG90

	set_servo_angle(SG90_CH, 0)
	sleep(0.5)
	set_servo_angle(SG90_CH, 180)
	sleep(0.5)

# Actuate 20kgcm servo

	set_servo_angle(SERVO20KG_CH, 0)
	sleep(0.5)
	set_servo_angle(SERVO20KG_CH, 5)
	sleep(2)
	set_servo_angle(SERVO20KG_CH, 0)
	sleep(0.5)

else:
	print("No Rust Detected")

cv2.imwrite("rust_detected.jpg", frame)
