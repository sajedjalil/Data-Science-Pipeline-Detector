#Reference: https://www.kaggle.com/yourwanghao/draper-satellite-image-chronology/align-images

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Any results you write to the current directory are saved as output.
def im_align_orb(imp1, imp2, nf=10000):
    """
    :param imp1: image1 file path
    :param imp2: image2 file path
    :param nf: max number of ORB key points
    :return:  transformed image2, so that it can be aligned with image1
    """
    img1 = cv2.imread(imp1, 0)
    img2 = cv2.imread(imp2, 0)
    h2, w2 = img2.shape[:2]

    orb = cv2.ORB_create(nfeatures=nf, WTA_K=2)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors.
    matches = bf.knnMatch(des1, des2, 2)

    # Sort them in the order of their distance.
    # matches_ = sorted(matches, key=lambda x: x.distance)[:5000]
    # print([m.distance for m in matches_])

    matches_ = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches_.append((m[0].trainIdx, m[0].queryIdx))

    #print("len(kp1), len(kp2), len(matches_)")

    kp1_ = np.float32([kp1[m[1]].pt for m in matches_]).reshape(-1, 1, 2)
    kp2_ = np.float32([kp2[m[0]].pt for m in matches_]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(kp2_, kp1_, cv2.RANSAC, 1.0)

    h1, w1 = img1.shape[:2]

    img2 = cv2.warpPerspective(cv2.imread(imp2), H, (w1, h1))
    return img2

def align_set_by_id(setid, isTrain=True, nFeatures=20000):
    """
    :param setid:
    :param isTrain:
    :return:
    """
    train_path = '../input/train_sm/'
    test_path = '../input/test_sm/'

    if isTrain == True:
        image_path = train_path
        fn1 = train_path + "set" + str(setid) + "_1.jpeg"
        outputpath = "./train_output"
    else:
        image_path = test_path
        fn1 = test_path + "set" + str(setid) + "_1.jpeg"
        outputpath = "./test_output/" 
    
    result=list()
    result.append(cv2.cvtColor(cv2.imread(fn1), cv2.COLOR_BGR2RGB))
    for id in [2, 3, 4, 5]:
        fn2 = image_path + "set" + str(setid) + "_" + str(id) + ".jpeg"
        print("fn1=%s, fn2=%s" % (os.path.basename(fn1), os.path.basename(fn2)))
        im = im_align_orb(fn1, fn2, nFeatures)
        #Note: kaggle script seems can't save output image? 
        #cv2.imwrite(outputpath + os.path.basename(fn2), im)
        result.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    #Note: kaggle script seems can't save output image? 
    #cv2.imwrite(outputpath + os.path.basename(fn1), cv2.imread(fn1))
    
    return result
    
from PIL import Image 

setimages=align_set_by_id(8, isTrain=False, nFeatures=15000)

plt.rcParams['figure.figsize'] = (16.0,16.0)

plt.subplot(321).set_title('image1'), plt.imshow(setimages[0]),plt.axis('off')
plt.subplot(323).set_title('image2'), plt.imshow(setimages[1]),plt.axis('off')
plt.subplot(324).set_title('image3'), plt.imshow(setimages[2]),plt.axis('off')
plt.subplot(325).set_title('image4'), plt.imshow(setimages[3]),plt.axis('off')
plt.subplot(326).set_title('image5'), plt.imshow(setimages[4]),plt.axis('off')

plt.show()