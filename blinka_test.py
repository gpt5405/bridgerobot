import board
import busio

i2c = busio.I2C(board.SCL, board.SDA)
print("I2C Initialized successfully.")

