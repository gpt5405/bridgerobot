import time
import adafruit_pca9685 import PCA9685

kit = ServoKit(channels=16)

kit.servo[0].angle = 180
kit.continuous_servo[1].throttle = 1
time.sleep(1)
kit.continuous_servo[1].thorttle = -1
time.sleep(1)
kit.servo[0].angle = 0
kit.continuous_servo[1].thorttle = 0
