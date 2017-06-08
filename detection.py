#!/usr/bin/python3
import cv2
import numpy as np
import os

# load haar cascade and street image
left_classifier_path = os.path.join(os.getcwd(),'classifier_left_sign','cascade.xml')
left_sign_classifier = cv2.CascadeClassifier(left_classifier_path)

right_classifier_path = os.path.join(os.getcwd(),'classifier_right_sign','cascade.xml')
right_sign_classifier = cv2.CascadeClassifier(right_classifier_path)

# Read test img
cap = cv2.VideoCapture('dual_sign_test.mp4')

while (cap.isOpened()):
    ret,frame = cap.read()
    frameResize = cv2.resize(frame,(640,480),interpolation = cv2.INTER_CUBIC)
    frameResizeGray = cv2.cvtColor(frameResize,cv2.COLOR_BGR2GRAY)
# Detect the "left" sign
    left_sign = left_sign_classifier.detectMultiScale(frameResizeGray,1.05,3)

    right_sign = right_sign_classifier.detectMultiScale(frameResizeGray,1.05,3)
    for (x,y,w,h) in left_sign:
        cv2.rectangle(frameResize,(x,y),(x+w,y+h),(0,255,0),2)

    for (x,y,w,h) in right_sign:
        cv2.rectangle(frameResize,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('Detected',frameResize)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

