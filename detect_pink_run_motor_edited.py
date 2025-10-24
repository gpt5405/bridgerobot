import cv2
import time
import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685

#Initialize I2C and PCA9685

i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50  #Standard for Servos

#Servo PWM Mapping

def angle_to_pwm(angle, pulse_min, pulse_max):
	pwm_min = int(pulse_min * pca.frequency / 1_000_000 * 4096)
	pwm_max = int(pulse_min * pca.frequency / 1_000_000 * 4096)
	return int(pwm_min + (angle / 180.0) * (pwm_max - pwm_min))

#Servo Configs

def move_servo(channel, angle): 
	if channel == 0: #SG90 Micro Servo

		pwm = angle_to_pwm(angle, pulse_min=500, pulse_max=2400)
	elif channel == 1:   #20kgcm Servo

		pwm = angle_to_pwm(angle, pulse_min=500, pulse_max=2500)
	else:
		return
	pca.channels[channel].duty_cycle = pwm

#Pink Detection Using HSV

def detect_pink(frame):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_pink = (140, 50, 50)
	upper_pink = (170, 255, 255)
	mask = cv2.inRange(hsv, lower_pink, upper_pink)
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	return any(cv2.contourArea(c) > 500 for c in contours)

# Main Loop

def main():
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("Camera not Found.")
		return
	try: 
		while True: 
			ret, frame = cap.read()
			if not ret:
				continue
			if detect_pink(frame):
				print("Pink Detected!")
				move_servo(0, 90)  # Move SG90 Servo 90 Degrees

				time.sleep(0.5)
				move_servo(1, 45) # 20kg Servo Move to 45 degrees

				time.sleep(0.5)
			else:
				move_servo(0, 0)
				move_servo(1, 0)

			cv2.imshow("Frame", frame)
			if cv2.waitkey(1):
				break
	finally:
		cap.release()
		cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
