# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import cv2
import tifffile as tiff

def _align_two_rasters(img1,img2):
    try:
        p1 = img1[0:3350,0:3338,1].astype(np.float32)
        p2 = img2[0:3350,0:3338,1].astype(np.float32)
    except:
        print("_align_two_rasters: can't extract patch, falling back to whole image")
        p1 = img1[:,:,1]
        p2 = img2[:,:,3]

    # lp1 = cv2.Laplacian(p1,cv2.CV_32F,ksize=5)
    # lp2 = cv2.Laplacian(p2,cv2.CV_32F,ksize=5)

    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-7)
    (cc, warp_matrix) = cv2.findTransformECC (p1, p2,warp_matrix, warp_mode, criteria)
    print("_align_two_rasters: cc:{}".format(cc))

    img3 = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    img3[img3 == 0] = np.average(img3)

    return img3


image_id = "6070_2_3"
img_3 = np.transpose(tiff.imread("../input/three_band/{}.tif".format(image_id)),(1,2,0))
img_a = np.transpose(tiff.imread("../input/sixteen_band/{}_M.tif".format(image_id)),(1,2,0))

raster_size = img_3.shape
print("Shape: ", img_3.shape)

img_a = cv2.resize(img_a,(raster_size[1],raster_size[0]),interpolation=cv2.INTER_CUBIC)

img_a_new = _align_two_rasters(img_3,img_a)

img_a = 255 * (img_a.astype(np.float32)-300) / (np.max(img_a) * 1.1) + 40
img_3 = 255 * img_3.astype(np.float32) / (np.max(img_3) * 0.9) + 60
img_a_new = 255 * (img_a_new.astype(np.float32)-300) / (np.max(img_a_new) * 1.1) + 40

#img_orig = np.stack([img_a[:, :, 2], img_a[:, :, 5], img_a[:, :, 6]], axis=-1).astype(np.uint8)
img_reg = np.stack([img_a_new[:, :, 2], img_a_new[:, :, 5], img_a_new[:, :, 6]], axis=-1).astype(np.uint8)
img_reg2 = np.stack([img_a_new[:, :, 5], img_a_new[:, :, 6], img_a_new[:, :, 7]], axis=-1).astype(np.uint8)
img_reg3 = np.stack([img_a_new[:, :, 0], img_a_new[:, :, 1], img_a_new[:, :, 2]], axis=-1).astype(np.uint8)
img_reg4 = np.stack([img_a_new[:, :, 3], img_a_new[:, :, 4], img_a_new[:, :, 5]], axis=-1).astype(np.uint8)

cv2.imwrite("registered3-7V1.tif",img_reg[0:3350,0:3338,:])
cv2.imwrite("registered6-8V1.png",img_reg2[0:3350,0:3338,:])
cv2.imwrite("registered1-3V1.png",img_reg3[0:3350,0:3338,:])
cv2.imwrite("registered4-6V1.png",img_reg4[0:3350,0:3338,:])
