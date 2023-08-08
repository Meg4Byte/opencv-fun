import cv2

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Create the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Apply the background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Show the original frame and the foreground mask
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
