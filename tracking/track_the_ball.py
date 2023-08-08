import cv2
import numpy as np

# Parameters for ball detection
ball_color_lower = np.array([0, 0, 200])  # Lower threshold for white color (adjust as needed)
ball_color_upper = np.array([180, 50, 255])  # Upper threshold for white color (adjust as needed)

# Initialize USB camera
cap = cv2.VideoCapture(0)

# Set up display windows
cv2.namedWindow("Camera Feed")
cv2.namedWindow("Tracked Ball")

while True:
    # Capture frame from camera
    ret, frame = cap.read()

    # Preprocess the frame
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, ball_color_lower, ball_color_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track the ball
    ball_detected = False
    centroid_x, centroid_y = 0, 0
    for contour in contours:
        # Filter contours based on area or other criteria if needed
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum contour area threshold
            ball_detected = True
            M = cv2.moments(contour)
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

    # Draw a green dot on the ball's centroid
    if ball_detected:
        cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

    # Display the frames in separate windows
    cv2.imshow('Camera Feed', frame)
    cv2.imshow('Tracked Ball', mask)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
