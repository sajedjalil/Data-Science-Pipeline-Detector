import os
import cv2
import numpy as np


img = cv2.imread("../input/train/83.png", cv2.IMREAD_GRAYSCALE)

kernel = np.ones((4,4), np.uint8) 
#eroding image, remove text, leave background
img_erode  = 255 - cv2.erode(255 - img, kernel,iterations = 1)
#subtract background
img_sub = cv2.add(img, - img_erode)
_, img_thresh = cv2.threshold(img_sub, 200, 255, cv2.THRESH_BINARY)


cv2.imwrite("0_original.png", img)
cv2.imwrite("1_background.png", img_erode)
cv2.imwrite("2_subtract_and_threshold.png", img_thresh)
