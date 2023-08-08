import cv2
import numpy as np

# Parameters for ball detection and color calibration
ball_color_hue_min = 0
ball_color_hue_max = 0
ball_color_saturation_min = 0
ball_color_saturation_max = 0
ball_color_value_min = 0
ball_color_value_max = 0

def on_trackbar_hue_min(value):
    global ball_color_hue_min
    ball_color_hue_min = value

def on_trackbar_hue_max(value):
    global ball_color_hue_max
    ball_color_hue_max = value

def on_trackbar_saturation_min(value):
    global ball_color_saturation_min
    ball_color_saturation_min = value

def on_trackbar_saturation_max(value):
    global ball_color_saturation_max
    ball_color_saturation_max = value

def on_trackbar_value_min(value):
    global ball_color_value_min
    ball_color_value_min = value

def on_trackbar_value_max(value):
    global ball_color_value_max
    ball_color_value_max = value

# Initialize USB camera
cap = cv2.VideoCapture(0)

# Create window for color calibration trackbars
cv2.namedWindow("Color Calibration")
cv2.createTrackbar("Hue Min", "Color Calibration", 0, 180, on_trackbar_hue_min)
cv2.createTrackbar("Hue Max", "Color Calibration", 0, 180, on_trackbar_hue_max)
cv2.createTrackbar("Saturation Min", "Color Calibration", 0, 255, on_trackbar_saturation_min)
cv2.createTrackbar("Saturation Max", "Color Calibration", 0, 255, on_trackbar_saturation_max)
cv2.createTrackbar("Value Min", "Color Calibration", 0, 255, on_trackbar_value_min)
cv2.createTrackbar("Value Max", "Color Calibration", 0, 255, on_trackbar_value_max)

while True:
    # Capture frame from camera
    ret, frame = cap.read()

    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current trackbar values for color thresholding
    hue_min_threshold = cv2.getTrackbarPos("Hue Min", "Color Calibration")
    hue_max_threshold = cv2.getTrackbarPos("Hue Max", "Color Calibration")
    saturation_min_threshold = cv2.getTrackbarPos("Saturation Min", "Color Calibration")
    saturation_max_threshold = cv2.getTrackbarPos("Saturation Max", "Color Calibration")
    value_min_threshold = cv2.getTrackbarPos("Value Min", "Color Calibration")
    value_max_threshold = cv2.getTrackbarPos("Value Max", "Color Calibration")

    # Set lower and upper threshold values for ball color detection
    ball_color_lower = np.array([ball_color_hue_min, ball_color_saturation_min, ball_color_value_min])
    ball_color_upper = np.array([ball_color_hue_max, ball_color_saturation_max, ball_color_value_max])

    # Apply color thresholding
    mask = cv2.inRange(hsv_frame, ball_color_lower, ball_color_upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display original frame and color thresholded result
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Color Thresholding Result", result)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Hue Min:", ball_color_hue_min)
        print("Hue Max:", ball_color_hue_max)
        print("Saturation Min:", ball_color_saturation_min)
        print("Saturation Max:", ball_color_saturation_max)
        print("Value Min:", ball_color_value_min)
        print("Value Max:", ball_color_value_max)
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()

"""
ue Min: 9
Hue Max: 25
Saturation Min: 146
Saturation Max: 255
Value Min: 113
Value Max: 255
"""
