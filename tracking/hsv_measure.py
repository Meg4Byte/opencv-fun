import cv2
import numpy as np

# Parameters for ball detection and color calibration
ball_color_hue = 0  # Initial value for ball color hue
ball_color_saturation = 0  # Initial value for ball color saturation
ball_color_value = 0  # Initial value for ball color value

def on_trackbar_hue(value):
    global ball_color_hue
    ball_color_hue = value

def on_trackbar_saturation(value):
    global ball_color_saturation
    ball_color_saturation = value

def on_trackbar_value(value):
    global ball_color_value
    ball_color_value = value

# Initialize USB camera
cap = cv2.VideoCapture(0)

# Create window for color calibration trackbars
cv2.namedWindow("Color Calibration")
cv2.createTrackbar("Hue", "Color Calibration", 0, 180, on_trackbar_hue)
cv2.createTrackbar("Saturation", "Color Calibration", 0, 255, on_trackbar_saturation)
cv2.createTrackbar("Value", "Color Calibration", 0, 255, on_trackbar_value)

while True:
    # Capture frame from camera
    ret, frame = cap.read()

    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current trackbar values for color thresholding
    hue_threshold = cv2.getTrackbarPos("Hue", "Color Calibration")
    saturation_threshold = cv2.getTrackbarPos("Saturation", "Color Calibration")
    value_threshold = cv2.getTrackbarPos("Value", "Color Calibration")

    # Set lower and upper threshold values for ball color detection
    ball_color_lower = np.array([ball_color_hue, ball_color_saturation, ball_color_value])
    ball_color_upper = np.array([ball_color_hue + hue_threshold, ball_color_saturation + saturation_threshold, ball_color_value + value_threshold])

    # Apply color thresholding
    mask = cv2.inRange(hsv_frame, ball_color_lower, ball_color_upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display original frame and color thresholded result
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Color Thresholding Result", result)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()

##R15 G113 B140