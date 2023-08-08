import cvzone
from cvzone.ColorModule import ColorFinder
import cv2


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
ct = ColorFinder(True)

# Custom Orange Color
hsvVals = {'hmin': 10, 'smin': 55, 'vmin': 215, 'hmax': 42, 'smax': 255, 'vmax': 255}
hsvRedVals = {'hmin': 0, 'smin': 100, 'vmin': 59, 'hmax': 55, 'smax': 204, 'vmax': 108}

while True:

    _ , frame = cap.read()


    imgColor, mask = ct.update(frame, hsvVals)

    imgContour , contour = cvzone.findContours(frame , mask)

    imgStack = cvzone.stackImages([frame, imgColor, mask , imgContour], 2, 1)

    cv2.imshow("Stacked Images", imgStack)
    ##cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()


 
"""
while True:

    success, img = cap.read()
    imgColor , mask = ct.update(img , hsvVals )
    lmList = ct.getPoints()

    if lmList:
        print(lmList)

    cv2.imshow("Image", img)
    imgStack = cvzone.stackImages([img, imgColor, mask], 2, 1)
    ##cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""
