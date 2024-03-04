import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # range of red
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    # range of blue
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # masking
    res_red = cv2.bitwise_and(frame, frame, mask=mask_red)
    res_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)

    # grayscale for countour
    gray_red = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)
    gray_blue = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)

    # detect circles
    circles_red = cv2.HoughCircles(gray_red, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles_blue = cv2.HoughCircles(gray_blue, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles_red is not None:
        circles_red = np.uint16(np.around(circles_red))
        for i in circles_red[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 255), 2)

    if circles_blue is not None:
        circles_blue = np.uint16(np.around(circles_blue))
        for i in circles_blue[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 0), 2)

    # result
    cv2.imshow('locations', frame)

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# goodbye
cap.release()
cv2.destroyAllWindows()
