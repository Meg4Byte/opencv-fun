import cv2
import numpy as np

# Initialize USB camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from camera
    ret, frame = cap.read()

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours
    cylinder_contour = None
    for contour in contours:
        # Adjust the filtering criteria based on your requirements
        area = cv2.contourArea(contour)
        if area > 1000:
            cylinder_contour = contour
            break

    # Draw edges on the frame
    if cylinder_contour is not None:
        cv2.drawContours(frame, [cylinder_contour], -1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Cylinder Edges', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
