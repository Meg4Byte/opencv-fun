from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

"""
    not really good for ball tracking , use the last one with single background

"""


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
"""Hue Min: 42
Hue Max: 78
Saturation Min: 0
Saturation Max: 194
Value Min: 0
Value Max: 112
"""
greenLower = (42, 0, 0)
greenUpper = (78, 194, 112)
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# Create the background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Duration for capturing background frames (in seconds)
bg_duration = 10
bg_frames = []
start_time = time.time()

# Capture frames for background averaging
while len(bg_frames) < bg_duration:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    bg_frames.append(mask)

    

# Calculate the average background mask
average_bg = np.mean(bg_frames, axis=0).astype(np.uint8)

# Keep looping
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

     # Apply background subtraction using the background subtractor object with the average background model
    fg_mask = bg_subtractor.apply(frame)
    
    # Threshold the foreground mask to obtain a binary image
    _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)

    mask_combined = cv2.bitwise_and(frame, frame, mask=fg_mask)  # Use fg_mask as a mask to retain moving objects
    
    # Convert mask_combined to grayscale
    mask_combined_gray = cv2.cvtColor(mask_combined, cv2.COLOR_BGR2GRAY)

    # Find contours in the combined mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask_combined_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # Only proceed if at least one contour was found
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)

        # Check if contour area is not zero before performing the division
        if cv2.contourArea(c) > 0:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Only proceed if the radius meets a minimum size
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

    pts.appendleft(center)

    # Loop over the set of tracked points
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Clean up
if not args.get("video", False):
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()
