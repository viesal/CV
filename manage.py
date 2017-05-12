# -*- coding: utf-8 -*-
import numpy as np
import cv2
import cv2.cv as cv
from datetime import datetime
import time

# #img = cv2.imread("1.jpeg")
cap1 = cv.CaptureFromCAM(-1)
cv.NamedWindow("capture", cv.CV_WINDOW_AUTOSIZE)
#
def CreateImg(capture, capture_name):
    img = cv.QueryFrame(capture)
    #cv.ShowImage(capture_name, img)
    #cv.WaitKey(10)
    path = u"img/cap{}.jpg".format(datetime.strftime(datetime.now(), "%S%f "))  # Уникальное имя для каждого кадра
    cv.SaveImage(path, img)
    return path

img = cv.LoadImage(CreateImg(cap1, 'capture'))
g = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 1)
b = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 1)
diff = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 3)
smoth = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 1)
hsv = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 3)
diff2 = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 3)


img1 = cv.QueryFrame(cap1)
time.sleep(5)
img2 = cv.QueryFrame(cap1)

cv.ShowImage('capture', img1)
cv.ShowImage('capture1', img2)
cv.WaitKey()

while 1:

    cv.Zero(g)
    cv.Zero(b)
    cv.Zero(diff)
    cv.Zero(smoth)
    cv.Zero(hsv)
    cv.Zero(diff2)
    print time.clock()

    img1 = cv.LoadImage(CreateImg(cap1, 'capture'))
    cv.ShowImage('img1', img1)
    cv.WaitKey()
    img2 = cv.LoadImage(CreateImg(cap1, 'capture'))
    cv.ShowImage('img2', img2)

    cv.Sub(img1, img2, diff)
    cv.ShowImage('capture1', diff)
    cv.CvtColor(diff, hsv, cv2.COLOR_BGR2HSV)

    lower_wite = cv.Scalar(50, 50, 50)
    uper_wite = cv.Scalar(255, 255, 255)

    frame_threshed = cv.CreateImage(cv.GetSize(hsv), 8, 1)
    cv.InRangeS(hsv, lower_wite, uper_wite, frame_threshed)
    cv.Copy(img2, diff2, frame_threshed)

    cv.ShowImage('capture', diff2)
    path = u"img/fin{}.jpg".format(datetime.strftime(datetime.now(), "%S%f "))  # Уникальное имя для каждого кадра
    cv.SaveImage(path, diff2)











