import cv2
import numpy as np

# Step 1: Camera Setup
cap = cv2.VideoCapture(0)  # Use the correct camera index if not 0

# Step 2: Color Detection
def detect_white_ball(frame):
    # Convert the frame to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    """
                    Hue Min: 98
                    Hue Max: 123
                    Saturation Min: 4
                    Saturation Max: 69
                    Value Min: 165
                    Value Max: 255


                    orange : 

                    orange_hue_min = 0
                    orange_hue_max = 26
                    orange_sat_min = 146
                    orange_sat_max = 255
                    orange_val_min = 113
                    orange_val_max = 255
    """
    
    # Define the lower and upper bounds of white color in HSV
    lower_white = np.array([0, 146, 113], dtype=np.uint8)
    upper_white = np.array([28, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only white color
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the white ball)
    if len(contours) > 0:
        ball_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(ball_contour)
        return int(x), int(y)
    else:
        return None

# Step 3: Kalman Filter Initialization
kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)

# Transition matrix (x, y, dx, dy)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)

# Measurement matrix (only x and y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], dtype=np.float32)

# Step 4: State Prediction
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=np.float32) * 0.03

# Step 5: Measurement Update
kalman.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], dtype=np.float32) * 0.1

# Initial state (x, y, dx, dy)
state = np.array([0, 0, 0, 0], dtype=np.float32)
kalman.statePre = state
kalman.statePost = state

# Step 6: Drawing the Results
while True:
    ret, frame = cap.read()

    if not ret:
        break

    ball_pos = detect_white_ball(frame)

    if ball_pos:
        # Step 4: State Prediction
        kalman.predict()

        # Step 5: Measurement Update
        measurement = np.array(ball_pos, dtype=np.float32)
        kalman.correct(measurement)

        # Get the corrected state (x, y, dx, dy)
        corrected_state = kalman.statePost

        # Draw the predicted position (x, y)
        predicted_pos = (int(corrected_state[0]), int(corrected_state[1]))
        cv2.circle(frame, predicted_pos, 10, (0, 255, 0), -1)

    # Draw the actual position of the ball (from color detection)
    if ball_pos:
        cv2.circle(frame, ball_pos, 5, (0, 0, 255), -1)

    cv2.imshow('Kalman Ball Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
