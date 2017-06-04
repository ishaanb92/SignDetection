#!/usr/bin/python3
import cv2
import numpy as np
import os

# load haar cascade and street image
left_classifier_path = os.path.join(os.getcwd(),'classifier_left_sign','cascade.xml')
left_sign_classifier = cv2.CascadeClassifier(left_classifier_path)

# Read test img
test_img = cv2.imread('blue.jpg')
test_img_resize = cv2.resize(test_img,(640,480),interpolation = cv2.INTER_CUBIC)
# Convert to grayscale
gray = cv2.cvtColor(test_img_resize,cv2.COLOR_BGR2GRAY)

# Detect the "left" sign
left_sign = left_sign_classifier.detectMultiScale(gray,1.05,3)

for (x,y,w,h) in left_sign:
    cv2.rectangle(test_img_resize,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('Detected',test_img_resize)
cv2.waitKey(0)
