# script to test if source files are OK
# seems that one / some of the files in the archive I downloaded was corrupted
# this kernel displays and allows to download the file as tiff file
# change array shape and transforms for 3band or non P 16bands


import sys, os
import numpy as np
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt

image_id = '6140_2_3'

testfile = os.path.join('..', 'input', 'sixteen_band', '{}_P.tif'.format(image_id))
try:
    img = tiff.imread(testfile)
    print("Opening image file: ", format(testfile))
    print("Image shape: ", img.shape)
except:
    print("Can't open image file: ", format(testfile))
    sys.exit("Ending script ...")



print(img[0, 0:255])

print("Image median: ", np.median(img))
print("Image minimum: ", np.min(img))
print("Image maximum: ", np.max(img))

# im_02 = np.transpose(tiff.imread(testfile), (0,1)).astype(np.uint16)
im_02 = np.transpose(img, (1,0)).astype(np.uint16)

print("Image median (2): ", np.median(im_02))
print("Image minimum (2): ", np.min(im_02))
print("Image maximum (2): ", np.max(im_02))

print(im_02[0, 0:255])

im_out = np.stack([im_02[:, :]], axis = -1).astype(np.uint16)
print("im_out shape: ", im_out.shape)

y1,y2,x1,x2 = 1000, 1600, 2000, 2600
region = im_out[y1:y2, x1:x2, 0]
plt.figure()
plt.imshow(region);

cv2.imwrite('{}_P_downloaded.tif'.format(image_id), im_out[0:im_out.shape[1], 0:im_out.shape[0], :])
# cv2.imwrite('{}_P_downloaded.png'.format(image_id), im_out[0:im_out.shape[1], 0:im_out.shape[0], :])