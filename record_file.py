# -*- coding: utf-8 -*-
import cv2
import numpy as np
import threading
import copy

camera = cv2.VideoCapture(0)

fourcc =  cv2.cv.CV_FOURCC(*'XVID')
video = cv2.VideoWriter('output3.avi', fourcc, 20, (640,480), 1)
while (camera.isOpened()):
    (grabbed, frame) = camera.read()
    if grabbed:
        video.write(frame)
        #frame = cv2.flip(frame, 0)
        cv2.imshow("masked", frame)
        key = cv2.waitKey(1) & 0xFF
    else:
        break
    if key == ord("q") or key == ord("Q"):
        break
camera.release()
video.release()
cv2.destroyAllWindows()