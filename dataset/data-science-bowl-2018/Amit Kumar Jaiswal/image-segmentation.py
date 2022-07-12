import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

import cv2
import matplotlib.pyplot as plt


def create_unified_mask(mask_image_paths):
    tmp_image_mask = None
    for m in mask_image_paths:
        m = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        if tmp_image_mask is None:
            tmp_image_mask = m
        tmp_image_mask = cv2.bitwise_or(tmp_image_mask, m)
    return tmp_image_mask

def get_image_type(image):
    # 0 is gray with black bg
    # 1 is gray with white/gray bg
    # 2 is colored

    image_type = -1
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v =cv2.split(hsv_image)
    
    # Decide if it is a colored image or not
    
    if np.max(h) == 0 and np.min(h) == 0:
        v_blurred = cv2.GaussianBlur(v, (5,5), 10)
        ret, thresh = cv2.threshold(v, 0, 255, cv2.THRESH_OTSU)
        _, cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        max_cnt_area = cv2.contourArea(cnts[0])
        
        # Decide which type of gray it is
        
        if max_cnt_area > 65000:
            image_type = 1
        else:
            image_type=0
    else:
        # TODO: here we can separate colored images based on the lightness of the BG. Just like we did it
        # for the gray images
        image_type = 2
    
    return image_type, (h, s, v)


train_df = pd.DataFrame()

train_image_ids = []
train_image_paths = []
train_image_mask_paths = []

for base_path in glob("../input/stage1_train/*"):
    image_id = os.path.basename(base_path)
    train_image_path = glob(os.path.join(base_path, "images", "*.png"))[0]
    mask_paths = glob(os.path.join(base_path, "masks", "*.png"))
    
    train_image_ids.append(image_id)
    train_image_paths.append(train_image_path)
    train_image_mask_paths.append(mask_paths)
    
train_df["image_id"] = train_image_ids
train_df["image_path"] = train_image_paths
train_df["mask_path"] = train_image_mask_paths


train_df.to_csv("train_df.csv")

test_df = pd.DataFrame()

test_image_ids = []
test_image_paths = []

for base_path in glob("../input/stage1_test/*"):
    image_id = os.path.basename(base_path)
    test_image_path = glob(os.path.join(base_path, "images", "*.png"))[0]
    
    test_image_ids.append(image_id)
    test_image_paths.append(test_image_path)
    
test_df["image_id"] = test_image_ids
test_df["image_path"] = test_image_paths


test_df.to_csv("test_df.csv")

img_type = [];
img_shape = [];
k = 0
while k < len(train_image_ids):
    img = cv2.imread(train_image_paths[k])

    img_mask = create_unified_mask(train_image_mask_paths[k])

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    image_type, (h, s, v) = get_image_type(img)
    img_shape.append(img.shape)
    img_type.append(image_type)
    k = k+1

train_df['Image_Type'] = img_type
train_df['ImageShape'] = img_shape

q = (train_df.Image_Type.value_counts())
w = (train_df.ImageShape.value_counts())
print('\nThere are %d GRAY with black bg images' % (q[0]))
print('There are %d GRAY with white/light-gray bg' % (q[1]))
print('There are %d COLOR with light bg\n' % (q[2]))
print(w)