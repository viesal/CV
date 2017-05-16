#!/usr/bin/env python
#  -*- coding: utf-8 -*-

import sys
import os
import argparse
from datetime import datetime
import imutils
import time
import cv2
import numpy as np
import threading


camera = cv2.VideoCapture(0)
back_frame = None
def save_img(path, img):
    cv2.imwrite(path, img)


# loop over the frames of the video
while True:
    #print datetime.strftime(datetime.now(), "%SS ")
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
    frame = imutils.resize(frame, width=500) # откусим от видеокадра 500 px
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # filtr = cv2.inRange(frame, np.array([0, 69, 255]), np.array([255, 255, 255]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 100, 250)
    edged1 = cv2.bitwise_and(frame, frame, mask=edged)

    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coins = frame.copy()


    cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)

    available_frame = frame.copy()
    if back_frame is not None:
        frameDelta = cv2.absdiff(available_frame, back_frame)
        cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY)

        # (B, G, R) = cv2.split(frameDelta)
        # (T, threshB) = cv2.threshold(B, 50, 255, cv2.THRESH_BINARY)
        # (T, threshG) = cv2.threshold(G, 50, 255, cv2.THRESH_BINARY)
        # (T, threshR) = cv2.threshold(R, 50, 255, cv2.THRESH_BINARY)
        # merge = cv2.merge([threshB, threshG, threshR])

        (T, threshAll) = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)
        merge = cv2.cvtColor(threshAll, cv2.COLOR_BGR2GRAY)
        # merge = cv2.cvtColor(merge, cv2.COLOR_BGR2GRAY)
        (T, thresh) = cv2.threshold(merge, 10, 255, cv2.THRESH_BINARY)
        masked = cv2.bitwise_and(available_frame, available_frame, mask=thresh)

        # Сохранение изображений при ненулевой маске
        # print np.sum(masked)
        # if np.sum(masked)>2550:
        #     path = u"img/mask{}.jpg".format(datetime.strftime(datetime.now(), "%S%f "))  # Уникальное имя для каждого кадра
        #     thr = threading.Thread(target=save_img, args=(path, masked))
        #     thr.deamon = True
        #     thr.start()

        cv2.imshow("masked", masked)
        cv2.imshow("coins", coins)
        cv2.imshow("edged", edged1)

        # cv2.imshow("gray", gray)
        # cv2.imshow("masked1", threshAll)
        # for width in thresh:
        #     for item in width:
        #         if item > 10:
        #             path = u"img/fin{}.jpg".format(datetime.strftime(datetime.now(), "%S%f "))  # Уникальное имя для каждого кадра
        #             cv2.imwrite(path, masked)

    back_frame = frame.copy()
    key = cv2.waitKey(1) & 0xFF
#
#         # if the `q` key is pressed, break from the lop
    if key == ord("q") or key == ord("Q"):
        # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()
