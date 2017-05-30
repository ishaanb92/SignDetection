#!/usr/bin/python3
import numpy as np
import cv2
import os

vidCapture = cv2.VideoCapture('neg_frames.avi')
# Read first frame
success, image = vidCapture.read()
print success
count = 0
while (vidCapture.isOpened()):
    success,image = vidCapture.read()
    if count >= 5000:
        break; # 5000 negative samples needed
    cv2.imwrite(os.path.join("neg_samples","frame{:d}.jpg".format(count)),image)
    count+=1

vidCapture.release()


