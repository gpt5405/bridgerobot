import cv2
import numpy as np

# step 1: Open the default camera (index 0)

cap = cv2.VideoCapture(0)

# Step 2: Capture one frame form the camera

ret, frame = cap.read()

# Step 3: Release the camera resource immediately

cap.release()

# Step 4: Check if the frame was successfully captured

if not ret:
	print("Failed to capture image")
	exit()

# Step 5: Convert the captured frame from BGR to HSV color space
# HSV isbetter for color detection because it seperates hue from brightness

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Step 6: Define the target pink color in RGB (213, 59, 119)
# Convert it to HSV for more accurate color matching

target_rgb = np.uint8([[[213, 59, 119]]]) # Shape (1,1,3)
target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV)[0][0] # Extract HSV Values

# Step 7: Create a range around the taarget HSV to allow for color variation

hue, sat, val = map(int, target_hsv) #Cast to int to prevent overflow
lower = np.array([max(hue - 10, 0), max(sat - 50, 0), max(val - 50, 0)])
upper = np.array([min(hue + 10, 179), min(sat + 50, 255), min(val + 50, 255)])

# Step 8: Create a binary maask where pinkn pixels fall within the HSV range

mask = cv2.inRange(hsv, lower, upper)

# Step 9: Find contours (connected regions) in the mask
# These representareas where pink is detected

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 10: Loop through each contour and draw bounding boxes around valid detections
# Filter out small areas to reduce noise

rust_found = False
for cnt in contours:
	area = cv2.contourArea(cnt)
	if area > 100: #Ignor tiny specks

		x, y, w, h = cv2.boundingRect(cnt)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green Box

		rust_found = True

# Step 11: Print detection result

if rust_found: 
	print("Rust Detected!")
else:
	print("No Rust Detected")
# Step 12: Save the annotated image to disk for review

cv2.imwrite("rust_detected.jpg", frame) 
