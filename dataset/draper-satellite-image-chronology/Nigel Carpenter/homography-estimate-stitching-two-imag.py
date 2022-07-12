##
##
## http://stackoverflow.com/questions/36796025/how-do-you-use-akaze-in-open-cv-on-python
##
##

import numpy as np
import cv2

from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

image1 = cv2.imread("../input/train_sm/set10_1.jpeg")
img1 = image1[800:1600, 800:1800] # Crop from centre third of mage: queryImage
img2 = cv2.imread("../input/train_sm/set10_2.jpeg") # train image

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)    

# initialize the AKAZE descriptor, then detect keypoints and extract
# local invariant descriptors from the image
detector = cv2.AKAZE_create()
(kp1, des1) = detector.detectAndCompute(gray1, None)
(kp2, des2) = detector.detectAndCompute(gray2, None)

print("keypoints: {}, descriptors: {}".format(len(kp1), des1.shape))
print("keypoints: {}, descriptors: {}".format(len(kp2), des2.shape))    


# Match the features
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des1,des1, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.95*n.distance:
        good.append([m])

if len(good)>MIN_MATCH_COUNT:
    print("Good Matches: {}".format(len(good)))
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[1:200], None, flags=2)
    cv2.imwrite('AKAZE matching.jpeg',img3)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None
    
