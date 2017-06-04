#!/usr/bin/python3
import cv2
import numpy as np

# Need thresholds on 3 colors
# Blue : Right, Left and Straight
# Red : Stop
# Yellow : U-Turn
def image_thresholding(img):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    smooth = cv2.GaussianBlur(img_hsv,(5,5),0)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # define range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    # define range of red color in HSV
    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask_red0 = cv2.inRange(img_hsv, lower_red, upper_red)
    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask_red1 = cv2.inRange(img_hsv, lower_red, upper_red)
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(smooth, lower_blue, upper_blue)
    # Threshold the HSV image to get only yellow colors
    mask_yellow = cv2.inRange(smooth, lower_yellow, upper_yellow)
    # Threshold the HSV image to get only red color
    mask_red = mask_red0 + mask_red1
    return mask_blue,mask_yellow,mask_red

def sign_detection_pipeline():
    img = cv2.imread('test.jpg')
    # Image resizing, for easier viewing
    resize_img = cv2.resize(img,(320,240), interpolation = cv2.INTER_CUBIC)
    # Get the thresholded images
    thresh_blue,thresh_yellow,thresh_red = image_thresholding(resize_img)
    cv2.imshow('Original',resize_img)
    cv2.imshow('Thresholded',thresh_blue)
    cv2.waitKey(0)

if __name__ == '__main__':
    sign_detection_pipeline()
