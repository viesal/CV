#!/usr/bin/env python
#  -*- coding: utf-8 -*-
import cv2
import imutils
import numpy as np
def nothing(x):
    pass
# camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture('output.avi')

cv2.namedWindow('hsv')
cv2.createTrackbar('h_min', 'hsv', 0, 255, nothing)
cv2.createTrackbar('s_min', 'hsv', 0, 255, nothing)
cv2.createTrackbar('v_min', 'hsv', 0, 255, nothing)
cv2.createTrackbar('h_max', 'hsv', 0, 255, nothing)
cv2.createTrackbar('s_max', 'hsv', 0, 255, nothing)
cv2.createTrackbar('v_max', 'hsv', 0, 255, nothing)
while True:
    (grabbed, frame) = camera.read()
    if not grabbed:  # проверка на захват камеры
        break
    frame = imutils.resize(frame, width=500)
    wite_min = np.array([cv2.getTrackbarPos('h_min','hsv'), cv2.getTrackbarPos('s_min','hsv'), cv2.getTrackbarPos('v_min','hsv')])
    wite_max = np.array([cv2.getTrackbarPos('h_max','hsv'), cv2.getTrackbarPos('s_max','hsv'), cv2.getTrackbarPos('v_max','hsv')])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # конвертируем фрейм из BGR в HSV
    range = cv2.inRange(hsv, wite_min, wite_max)
    cv2.imshow("hsv", range)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("p") or key == ord("P"):
        cv2.waitKey()
    if key == ord("q") or key == ord("Q"):
        camera.release()
        cv2.destroyAllWindows()