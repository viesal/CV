#!/usr/bin/env python
#  -*- coding: utf-8 -*-

import sys
import os
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np


camera = cv2.VideoCapture(0)
back_frame = None



# loop over the frames of the video
while True:
    # time.sleep(0.5)
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
    frame = imutils.resize(frame, width=500)  # откусим от видеокадра 500 px
    gray = frame.copy()
    if back_frame is not None:
        frameDelta = cv2.absdiff(gray, back_frame)
        cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY)
        (B, G, R) = cv2.split(frameDelta)
        (T, threshB) = cv2.threshold(B, 50, 255, cv2.THRESH_BINARY)
        (T, threshG) = cv2.threshold(G, 50, 255, cv2.THRESH_BINARY)
        (T, threshR) = cv2.threshold(R, 50, 255, cv2.THRESH_BINARY)
        merge = cv2.merge([threshB, threshG, threshR])
        merge = cv2.cvtColor(merge, cv2.COLOR_BGR2GRAY)
        (T, thresh) = cv2.threshold(merge, 10, 255, cv2.THRESH_BINARY)
        masked = cv2.bitwise_and(gray, gray, mask=thresh)
        print masked.__
        cv2.imshow("masked", masked)

    back_frame = frame.copy()
    key = cv2.waitKey(1) & 0xFF
#
#         # if the `q` key is pressed, break from the lop
    if key == ord("q") or key == ord("Q"):
        # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()
