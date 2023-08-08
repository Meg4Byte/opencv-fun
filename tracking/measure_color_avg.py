import cv2
import numpy as np

# Camera Setup
cap = cv2.VideoCapture(0)

# Capture the Image
ret, frame = cap.read()

# Convert to HSV Color Space
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define the White Color Range (example values)
lower_white = np.array([98, 4, 165], dtype=np.uint8)
upper_white = np.array([123, 69, 255], dtype=np.uint8)


# Thresholding
white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)

# Extract White Color
white_color = cv2.bitwise_and(hsv_frame, hsv_frame, mask=white_mask)

# Convert back to BGR color space for visualization
white_color_bgr = cv2.cvtColor(white_color, cv2.COLOR_HSV2BGR)

# Show the original image and the extracted white color
cv2.imshow('Original Image', frame)
cv2.imshow('White Color', white_color_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()
