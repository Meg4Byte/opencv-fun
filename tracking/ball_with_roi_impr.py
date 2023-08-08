from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("--roi_x", type=int, default=200,
    help="X-coordinate of the top-left corner of the ROI")
ap.add_argument("--roi_y", type=int, default=100,
    help="Y-coordinate of the top-left corner of the ROI")
ap.add_argument("--roi_w", type=int, default=300,
    help="Width of the ROI")
ap.add_argument("--roi_h", type=int, default=300,
    help="Height of the ROI")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

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

# Define the adaptive color thresholding parameters
""""
Saturation Min: 64
Saturation Max: 190
Value Min: 44
Value Max: 118
"""
learning_rate = 0.5
min_saturation = 64
max_saturation = 190
min_value = 44
max_value = 100

# Keep looping
start_time = None
travel_distance = 0
ball_detected = False

while True:
    # grab the current frame
    frame = vs.read()

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # Draw ROI on the frame
    cv2.rectangle(frame, (args["roi_x"], args["roi_y"]), (args["roi_x"] + args["roi_w"], args["roi_y"] + args["roi_h"]), (0, 255, 0), 2)

    # Extract the ROI from the frame
    roi = frame[args["roi_y"]:args["roi_y"]+args["roi_h"], args["roi_x"]:args["roi_x"]+args["roi_w"]]

    # resize the ROI and convert it to the HSV color space
    roi = imutils.resize(roi, width=600)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # apply background subtraction to update the adaptive color thresholding model
    bg_subtractor.apply(roi, learningRate=learning_rate)

    # get the current model of the background without the ball
    bg_model = bg_subtractor.getBackgroundImage()

    # dynamically adjust the color threshold based on the current background model
    mask = cv2.inRange(hsv, (0, min_saturation, min_value), (255, max_saturation, max_value))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(bg_model))

    # find contours in the mask and initialize the current (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        
        # Check if contour area is not zero before performing the division
        if cv2.contourArea(c) > 0:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 5:
                # Draw the circle and centroid on the ROI,
                # then update the list of tracked points
                cv2.circle(roi, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(roi, center, 2, (0, 0, 255), -1)

                # Calculate the distance traveled by the ball
                if pts and pts[-1] is not None:
                    travel_distance += np.sqrt((center[0] - pts[-1][0]) ** 2 + (center[1] - pts[-1][1]) ** 2)

                # Check if the ball center passes through the upper and lower bounds of the ROI
                if center[1] <= args["roi_y"] and pts and pts[-1] and pts[-1][1] > args["roi_y"] and not ball_detected:
                    # The ball has started moving
                    start_time = time.time()
                    ball_detected = True

                if center[1] >= args["roi_y"] + args["roi_h"] and pts and pts[-1] and pts[-1][1] < args["roi_y"] + args["roi_h"]:
                    # The ball has finished moving
                    if start_time:
                        end_time = time.time()
                        time_taken = end_time - start_time
                        print("Time taken:", time_taken, "seconds")
                        start_time = None
                        travel_distance = 0
                        ball_detected = False

    # Resize the ROI back to its original size and assign it back to the frame
    roi = imutils.resize(roi, width=args["roi_w"])
    frame[args["roi_y"]:args["roi_y"]+args["roi_h"], args["roi_x"]:args["roi_x"]+args["roi_w"]] = roi

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

    # update the points queue
    pts.appendleft(center)

# Clean up
if not args.get("video", False):
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()
