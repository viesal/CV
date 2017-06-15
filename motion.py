#  -*- coding: utf-8 -*-
#!/usr/bin/env python


import sys
import os
import argparse
from datetime import datetime

import imutils
import time
import cv2
import numpy as np
import threading
import copy

camera = cv2.VideoCapture('output1.avi')
# camera = cv2.VideoCapture(0)

class Timer():
    def __init__(self):
        self.time_start = datetime.now()
    def get_time(self):
        delta = datetime.now()-self.time_start
        return delta.microseconds / 100000.00


def framing(frame, color_min, color_max):
    """
    Передаем в функцию кадр и границы вета в формате hsv
    конвертируем фрейм из BGR в HSV
    выделяем маску выбранного цвета
    находим на маске все контуры
    выбираем максимальный по площади контур
    определеяем координаты максимального описывающего прямоугольника
    :param frame:
    :param color_min:
    :param color_max:
    :return:    x , y - координаты левого верхнего угла прямоугольника
                w - ширина
                h - высота
    """
    max = np.array([(0, 0), (0, 0)], np.float32)
    cv2.isContourConvex(max)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    range = cv2.inRange(hsv, color_min, color_max)
    (cnts, _) = cv2.findContours(range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in cnts]
    max_index = np.argmax(areas)
    cnt = cnts[max_index]
    x, y, w, h = cv2.boundingRect(cnt)
    return  x+10, y+10, w-20, h-20

def save_img_in_thread(prefix, img):
    """
    Сохраниени изображений в потоке
    :param prefix: префикс в названии изображения
    :param img: изображение
    :return:
    """

    path = u"img1/{}{}.jpg".format(prefix, datetime.strftime(datetime.now(), "%H_%M_%S_%f"))  # Уникальное имя для каждого кадра
    thr = threading.Thread(target=cv2.imwrite, args=(path, img))
    thr.deamon = True
    thr.start()

iter1 = True
iter2 = True
back_frame = None
first_frame = None
purpur_min = np.array([112, 0, 0])
purpur_max = np.array([159, 255, 255])
blue_min = np.array([85, 87, 43])
blue_max = np.array([102, 194, 93])

shin_PI = cv2.imread('img1/roi04_09_18_522548.jpg', 0)
start_AC = cv2.imread('img1/roi04_09_17_466423.jpg', 0)

# loop over the frames of the video
while True:
    time.sleep(0.001)
    (grabbed, frame) = camera.read()                                                # захват фрейма
    if not grabbed:                                                                 # проверка на захват камеры
        break
    cv2.imshow("frame", frame)
    #frame = imutils.resize(frame, width=500)                                        # откусим от видеокадра 500 px
    height_frame, width_frame, channels = frame.shape                               # определяем размеры фрейма
    one_frame = frame.copy()
    if first_frame is None:                                                         # захватываем первый фрейм
        first_frame = frame.copy()
    x_ff, y_ff, w_ff, h_ff = framing(first_frame, blue_min, blue_max)
    rectangle = np.zeros((height_frame, width_frame), dtype="uint8")
    cv2.rectangle(rectangle, (x_ff, y_ff), (x_ff + w_ff, y_ff + h_ff), 255, -1)     # по координатам создаем маску
    available_frame = cv2.bitwise_and(frame, frame, mask=rectangle)                 # накладываем маску на начальный кадр
    coins = frame.copy()                                                            # копируем текущий фрейм
    cv2.rectangle(coins, (x_ff, y_ff), (x_ff + w_ff, y_ff + h_ff), (0, 255, 0), 2)  # по координатам рисуем прямоугольник
    if back_frame is not None:
        back_frame = cv2.bitwise_and(back_frame, back_frame, mask=rectangle)
        frameDelta = cv2.absdiff(available_frame, back_frame)
        gray = cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY)
        (T, threshAll) = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        cv2.imshow("threshAll", threshAll)
        if np.sum(threshAll) > 1250:
            (cnts_t, _) = cv2.findContours(threshAll, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            areas = [cv2.contourArea(c) for c in cnts_t]
            a = available_frame.copy()
            cv2.drawContours(a, cnts_t, -1, (0, 255, 0), 2)
            cv2.imshow("a", a)
            # max_t = np.array([(0, 0), (0, 0)], np.float32)
            # cv2.isContourConvex(max_t)
            # areas = [cv2.contourArea(c) for c in cnts_t]
            # max_index = np.argmax(areas)
            # cnt = cnts_t[max_index]
            for cnt in cnts_t:
                x_t, y_t, w_t, h_t = cv2.boundingRect(cnt)
            #der = available_frame.copy()
            # cv2.rectangle(available_frame, (x_t, y_t), (x_t+w_t, y_t+h_t), (0, 255, 0), 2)
                roi = available_frame[y_t:(y_t + h_t), x_t:(x_t + w_t)]
            # cv2.imshow("available_frame", available_frame)
                #save_img_in_thread('roi', roi)

            # roi_gray = cv2.cvtColor(available_frame, cv2.COLOR_BGR2GRAY)
            # w_pi, h_pi = shin_PI.shape[::-1]
            # w_ac, h_ac = start_AC.shape[::-1]
            #
            # res1 = cv2.matchTemplate(roi_gray, shin_PI, cv2.TM_CCOEFF_NORMED)
            # #print(res)
            # loc1 = np.where(res1 >= 0.80)
            # #print np.size(loc)
            # if np.size(loc1) is not 0 and iter1 is True:
            #     print("Включение шины ПИ ", timer.get_time())
            #     iter1 = False
            #
            #     # for pt in zip(*loc1[::-1]):
            #     #     cv2.rectangle(available_frame, pt, (pt[0] + w_pi, pt[1] + h_pi), (0, 0, 255), 2)
            #     # cv2.imshow("shin_PI", available_frame)
            # loc1 = None
            #
            # res2 = cv2.matchTemplate(roi_gray, start_AC, cv2.TM_CCOEFF_NORMED)
            # # print(res)
            # loc2 = np.where(res2 >= 0.9)
            # # print np.size(loc)
            # if np.size(loc2) is not 0 and iter2 is True:
            #     timer = Timer()
            #     print(u"начало АЦ ", timer.get_time())
            #     iter2 = False
            #
            #     # for pt in zip(*loc2[::-1]):
            #     #     cv2.rectangle(available_frame, pt, (pt[0] + w_ac, pt[1] + h_ac), (0, 0, 255), 2)
            #     # cv2.imshow("start_AC", available_frame)
            # loc2 = None

            # res3 = cv2.matchTemplate(roi_gray, start_AC, cv2.TM_CCOEFF_NORMED)
            # # print(res)
            # loc2 = np.where(res2 >= 0.9)
            # # print np.size(loc)
            # if np.size(loc2) is not 0 and iter2 is True:
            #     timer = Timer()
            #     print(u"начало АЦ ", timer.get_time())
            #     iter2 = False
            #
            #     # for pt in zip(*loc2[::-1]):
            #     #     cv2.rectangle(available_frame, pt, (pt[0] + w_ac, pt[1] + h_ac), (0, 0, 255), 2)
            #     # cv2.imshow("start_AC", available_frame)
            # loc2 = None

        #cv2.imshow("frame", frame)


        #masked = cv2.bitwise_and(available_frame, available_frame, mask=threshAll)
        # cv2.imshow("available_frame", available_frame)
        #cv2.imshow("available_frame", available_frame)

            # res = cv2.matchTemplate(roi_gray, shin_S, cv2.TM_CCOEFF_NORMED)

            #save_img_in_thread('mask', masked)


        # x_f, y_f, w_f, h_f = framing(one_frame, purpur_min, purpur_max)
        # cv2.rectangle(one_frame, (x_f, y_f), (x_f + w_f, y_f + h_f), (0, 255, 0), 2)
        # cv2.imshow("masked", one_frame)

        # cv2.imshow("back_frame", der)

        #cv2.imshow("masked", masked)

    back_frame = frame.copy()
    key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
    if key == ord("p") or key == ord("P"):
        cv2.waitKey()
    if key == ord("q") or key == ord("Q"):
        # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()
